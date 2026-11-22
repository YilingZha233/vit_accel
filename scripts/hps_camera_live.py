#!/usr/bin/env python3
"""Windows viewer: HPS USB capture + persistent HPS ncnn runner over SSH.

Windows performs Pillow RGB preprocessing and display, never model inference.
See docs/persistent_camera_demo.md. Hardware validation is still required.
"""
import argparse
import csv
import struct
import getpass
import io
import shlex
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


def preprocess(image):
    image = image.convert('RGB')
    w, h = image.size
    size = (256, int(256 * h / w)) if w <= h else (int(256 * w / h), 256)
    image = image.resize(size, Image.Resampling.BICUBIC)
    left, top = round((size[0] - 224) / 2), round((size[1] - 224) / 2)
    crop = image.crop((left, top, left + 224, top + 224))
    x = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    x = (x - np.array([.485, .456, .406], dtype=np.float32)[:, None, None])
    x /= np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    return crop, x.astype('<f4').tobytes()


class PersistentSession:
    """One request in flight. Drain diagnostics and enforce whole-operation deadlines."""
    def __init__(self, channel, timeout):
        self.channel, self.timeout = channel, timeout
        self.errors = bytearray()
        self.load_ms = None

    def diagnostics(self):
        while self.channel.recv_stderr_ready():
            self.errors.extend(self.channel.recv_stderr(4096))
            self.errors = self.errors[-8192:]

    def receive(self, size, deadline):
        output = bytearray()
        while len(output) < size:
            self.diagnostics()
            if time.monotonic() >= deadline:
                raise TimeoutError('HPS response timeout. ' + self.errors.decode(errors='replace'))
            if self.channel.recv_ready():
                chunk = self.channel.recv(size - len(output))
                if not chunk:
                    raise RuntimeError('HPS closed stream. ' + self.errors.decode(errors='replace'))
                output.extend(chunk)
            elif self.channel.exit_status_ready() or self.channel.closed:
                self.diagnostics()
                raise RuntimeError('HPS runner stopped. ' + self.errors.decode(errors='replace'))
            else:
                time.sleep(.002)
        return bytes(output)

    def ready(self):
        magic, self.load_ms = struct.unpack('<4sd', self.receive(12, time.monotonic() + self.timeout))
        if magic != b'RDY1' or not np.isfinite(self.load_ms) or self.load_ms < 0:
            raise RuntimeError('Invalid ready message; use camera_runner, not model_runner')

    def predict(self, frame_id, raw):
        if len(raw) != 602112:
            raise ValueError('Expected 602112 input bytes')
        deadline = time.monotonic() + self.timeout
        message = struct.pack('<4sI', b'VIT1', frame_id) + raw
        offset = 0
        while offset < len(message):
            self.diagnostics()
            if time.monotonic() >= deadline:
                raise TimeoutError('Tensor send timeout')
            if self.channel.closed or self.channel.exit_status_ready():
                raise RuntimeError('HPS runner stopped. ' + self.errors.decode(errors='replace'))
            if self.channel.send_ready():
                n = self.channel.send(message[offset:offset + 65536])
                if not n:
                    raise RuntimeError('HPS closed input')
                offset += n
            else:
                time.sleep(.002)
        header = self.receive(32, deadline)
        magic, returned_id, pack_ms, infer_ms, service_ms = struct.unpack('<4sIddd', header)
        if magic != b'RES1' or returned_id != frame_id:
            raise RuntimeError('Mismatched HPS response/frame ID')
        if not all(np.isfinite(t) and t >= 0 for t in (pack_ms, infer_ms, service_ms)):
            raise RuntimeError('Invalid HPS timing')
        output = self.receive(4000, deadline)
        logits = np.frombuffer(output, dtype='<f4')
        if not np.isfinite(logits).all():
            raise RuntimeError('Nonfinite HPS logits')
        return logits, pack_ms, infer_ms, service_ms


def runner_command(args):
    return 'exec ' + ' '.join(map(shlex.quote, [args.runner, args.param, args.model,
        str(args.threads), args.input_blob, args.output_blob, args.precision]))


def main():
    import paramiko
    from PIL import ImageTk
    import tkinter as tk

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--user', default='root')
    p.add_argument('--password', action='store_true', help='Prompt without echo')
    p.add_argument('--known-hosts', default=str(Path.home() / '.ssh' / 'known_hosts'))
    p.add_argument('--ffmpeg', default='ffmpeg')
    p.add_argument('--device', default='/dev/video0')
    p.add_argument('--size', default='640x480')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--format', choices=['mjpeg', 'yuyv422'], default='mjpeg')
    p.add_argument('--runner', required=True)
    p.add_argument('--param', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--precision', choices=['fp32', 'runtime-int8'], default='runtime-int8')
    p.add_argument('--threads', type=int, choices=[1, 2], default=2)
    p.add_argument('--input-blob', default='in0')
    p.add_argument('--output-blob', default='out0')
    p.add_argument('--labels', type=Path, help='1000 UTF-8 lines in checkpoint index order')
    p.add_argument('--image', type=Path, help='Classify one saved image instead of starting the camera')
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--preview-hz', type=int, default=20)
    p.add_argument('--timings', type=Path, help='New CSV path; refuses to overwrite')
    args = p.parse_args()
    labels = args.labels.read_text(encoding='utf-8').splitlines() if args.labels else None
    if labels is not None and len(labels) != 1000:
        p.error('--labels must contain exactly 1000 lines, index 0 first')
    if args.fps <= 0 or args.timeout <= 0 or not 1 <= args.preview_hz <= 60:
        p.error('fps/timeout must be positive and preview-hz must be 1..60')
    if args.timings and args.timings.exists():
        p.error('--timings file exists; choose a new filename')
    client = paramiko.SSHClient()
    client.load_host_keys(args.known_hosts)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, username=args.user,
                   password=getpass.getpass('Board SSH password: ') if args.password else None,
                   timeout=15, auth_timeout=30, banner_timeout=30)
    client.get_transport().set_keepalive(15)
    stop = threading.Event()
    lock = threading.Lock()
    state = dict(latest=None, result=None, error='', busy=False, count=0, load_ms=None)

    def fail(message):
        with lock:
            state['error'] = message
        stop.set()

    def capture():
        channel = None
        try:
            cmd = [args.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
                   '-f', 'v4l2', '-input_format', args.format, '-video_size', args.size,
                   '-framerate', str(args.fps), '-i', args.device, '-an']
            cmd += ['-c:v', 'copy'] if args.format == 'mjpeg' else ['-c:v', 'mjpeg', '-threads', '1', '-q:v', '5']
            cmd += ['-f', 'image2pipe', 'pipe:1']
            channel = client.get_transport().open_session(timeout=15)
            channel.settimeout(1)
            channel.exec_command('exec ' + ' '.join(map(shlex.quote, cmd)))
            buf, errors = bytearray(), bytearray()
            last = time.monotonic()
            while not stop.is_set():
                if channel.recv_stderr_ready():
                    errors.extend(channel.recv_stderr(4096))
                    errors = errors[-8192:]
                if not channel.recv_ready():
                    if channel.exit_status_ready():
                        raise RuntimeError('Capture stopped: ' + errors.decode(errors='replace'))
                    if time.monotonic() - last > 15:
                        raise RuntimeError('No camera frame for 15 seconds. ' + errors.decode(errors='replace'))
                    stop.wait(.01)
                    continue
                data = channel.recv(65536)
                if not data:
                    raise RuntimeError('Camera stream closed')
                buf.extend(data)
                if len(buf) > 8 * 1024 * 1024:
                    raise RuntimeError('Invalid or oversized JPEG stream')
                while True:
                    start = buf.find(b'\xff\xd8')
                    if start < 0:
                        buf[:] = buf[-1:]
                        break
                    end = buf.find(b'\xff\xd9', start + 2)
                    if end < 0:
                        del buf[:start]
                        break
                    jpeg = bytes(buf[start:end + 2])
                    del buf[:end + 2]
                    last = time.monotonic()
                    with lock:
                        state['count'] += 1
                        state['latest'] = (state['count'], last, jpeg)
        except Exception as exc:
            if not stop.is_set():
                fail(str(exc))
        finally:
            if channel is not None:
                channel.close()

    def classify():
        previous = -1
        channel = None
        log = None
        try:
            channel = client.get_transport().open_session(timeout=15)
            channel.settimeout(1)
            channel.exec_command(runner_command(args))
            session = PersistentSession(channel, args.timeout)
            session.ready()
            with lock:
                state['load_ms'] = session.load_ms
            fields = ['frame_id', 'first_request', 'decode_ms', 'preprocess_ms', 'roundtrip_ms',
                      'pack_ms', 'infer_ms', 'service_ms', 'transport_overhead_ms',
                      'request_ms', 'received_to_result_ms', 'load_ms', 'top1']
            writer = None
            if args.timings:
                log = args.timings.open('x', newline='', encoding='utf-8')
                writer = csv.DictWriter(log, fieldnames=fields)
                writer.writeheader()
            first = True
            while not stop.is_set():
                with lock:
                    item = state['latest']
                if item is None or item[0] == previous:
                    stop.wait(.05)
                    continue
                frame_id, received, jpeg = item
                previous = frame_id
                with lock:
                    state['busy'] = True
                started = time.monotonic()
                frame = Image.open(io.BytesIO(jpeg)).convert('RGB')
                decoded = time.monotonic()
                crop, raw = preprocess(frame)
                preprocessed = time.monotonic()
                logits, pack_ms, infer_ms, service_ms = session.predict(frame_id, raw)
                completed = time.monotonic()
                roundtrip_ms = (completed - preprocessed) * 1000
                metrics = dict(frame_id=frame_id, first_request=int(first),
                    decode_ms=(decoded-started)*1000, preprocess_ms=(preprocessed-decoded)*1000,
                    roundtrip_ms=roundtrip_ms, pack_ms=pack_ms, infer_ms=infer_ms, service_ms=service_ms,
                    transport_overhead_ms=max(0, roundtrip_ms-service_ms),
                    request_ms=(completed-started)*1000, received_to_result_ms=(completed-received)*1000,
                    load_ms=session.load_ms, top1=int(np.argmax(logits)))
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                top = np.argsort(-logits)[:5]
                lines = [f'{labels[i] if labels else "class " + str(i)}: {probabilities[i]:.1%}' for i in top]
                if writer:
                    writer.writerow(metrics)
                    log.flush()
                with lock:
                    state['result'] = (frame_id, received, crop, '\n'.join(lines), metrics)
                    state['busy'] = False
                first = False
        except Exception as exc:
            if not stop.is_set():
                fail(str(exc))
        finally:
            if log:
                log.close()
            if channel is not None:
                channel.close()

    root = tk.Tk()
    root.title('DE1-SoC camera | HPS-only TinyViT inference')
    tk.Label(root, text='Live camera preview').grid(row=0, column=0)
    tk.Label(root, text='Last classified 224 x 224 crop').grid(row=0, column=1)
    live, classified = tk.Label(root), tk.Label(root)
    live.grid(row=1, column=0)
    classified.grid(row=1, column=1)
    info = tk.Label(root, justify='left', wraplength=850)
    info.grid(row=2, column=0, columnspan=2, sticky='w')
    tk.Label(root, text='Windows: preprocessing/display. HPS: capture/model. Scores: softmax, not calibrated confidence.').grid(row=3, column=0, columnspan=2)

    rendered = {'live': -1, 'result': -1}

    def refresh():
        with lock:
            latest, result, error, busy, load_ms = (state[k] for k in ('latest', 'result', 'error', 'busy', 'load_ms'))
        text = 'Loading model once / waiting for camera...'
        if latest:
            if latest[0] != rendered['live']:
                try:
                    image = Image.open(io.BytesIO(latest[2])).convert('RGB')
                    image.thumbnail((640, 480))
                    live.image = ImageTk.PhotoImage(image)
                    live.configure(image=live.image)
                    rendered['live'] = latest[0]
                except Exception as exc:
                    fail('JPEG preview decode failed: ' + str(exc))
            text = f'Preview frame {latest[0]}; received {time.monotonic() - latest[1]:.1f}s ago. '
        if load_ms is not None:
            text += f' Model loaded once: {load_ms:.0f} ms.'
        if result:
            frame_id, received, crop, scores, m = result
            if frame_id != rendered['result']:
                classified.image = ImageTk.PhotoImage(crop)
                classified.configure(image=classified.image)
                rendered['result'] = frame_id
            text += (f'\nResult for frame {frame_id}; image received {time.monotonic() - received:.1f}s ago.'
                     f' {"First request (cold)." if m["first_request"] else ""}'
                     f'\nJPEG decode {m["decode_ms"]:.1f} ms | Preprocess {m["preprocess_ms"]:.1f} ms'
                     f' | HPS inference {m["infer_ms"]:.0f} ms'
                     f'\nTransfer/IPC overhead estimate {m["transport_overhead_ms"]:.0f} ms'
                     f' | Total request {m["request_ms"]/1000:.2f} s (one inference)\n{scores}')
        text += '\nClassifying newest available frame...' if busy else ''
        if error:
            text = 'STOPPED / STALE — restart after fixing:\n' + error + '\n' + text
        info.configure(text=text)
        root.after(max(1, 1000 // args.preview_hz), refresh)

    def close():
        stop.set()
        client.close()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', close)
    if args.image:
        state['latest'] = (1, time.monotonic(), args.image.read_bytes())
    else:
        threading.Thread(target=capture, daemon=True).start()
    threading.Thread(target=classify, daemon=True).start()
    refresh()
    try:
        root.mainloop()
    finally:
        stop.set()
        client.close()


if __name__ == '__main__':
    main()
