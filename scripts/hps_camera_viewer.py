#!/usr/bin/env python3
"""Windows viewer: HPS USB capture + existing HPS ncnn runner over SSH.

Windows performs Pillow RGB preprocessing and display, never model inference.
See docs/usb_camera_hps_windows_guide.md. Hardware validation is still required.
"""
import argparse
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


def inference_command(args):
    # Only this invocation's mktemp directory is removed; model files are read-only.
    q = shlex.quote
    runner = ' '.join(map(q, [args.runner, 'eval', args.param, args.model]))
    tail = ' '.join(map(q, [str(args.threads), '0', '1', args.input_blob,
                            args.output_blob, '0', args.precision]))
    return (
        'set -eu; job=$(mktemp -d /tmp/vitcam.XXXXXX); '
        'trap \'rm -rf "$job"\' EXIT; '
        'cat > "$job/input.bin"; '
        'test "$(wc -c < "$job/input.bin")" -eq 602112; '
        'printf "camera\\t0\\tinput.bin\\n" > "$job/inputs.tsv"; '
        f'{runner} "$job/inputs.tsv" "$job/result" {tail} >&2; '
        'cat "$job/result/00000.logits.bin"'
    )


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
    p.add_argument('--fps', type=int, default=5)
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
    args = p.parse_args()
    labels = args.labels.read_text(encoding='utf-8').splitlines() if args.labels else None
    if labels is not None and len(labels) != 1000:
        p.error('--labels must contain exactly 1000 lines, index 0 first')
    if args.fps <= 0 or args.timeout <= 0:
        p.error('fps and timeout must be positive')
    client = paramiko.SSHClient()
    client.load_host_keys(args.known_hosts)
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, username=args.user,
                   password=getpass.getpass('Board SSH password: ') if args.password else None,
                   timeout=15, auth_timeout=30, banner_timeout=30)
    client.get_transport().set_keepalive(15)
    stop = threading.Event()
    lock = threading.Lock()
    state = dict(latest=None, result=None, error='', busy=False, count=0)

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
                    frame = Image.open(io.BytesIO(jpeg)).convert('RGB')
                    last = time.monotonic()
                    with lock:
                        state['count'] += 1
                        state['latest'] = (state['count'], last, frame)
        except Exception as exc:
            if not stop.is_set():
                fail(str(exc))
        finally:
            if channel is not None:
                channel.close()

    def classify():
        previous = -1
        try:
            while not stop.is_set():
                with lock:
                    item = state['latest']
                if item is None or item[0] == previous:
                    stop.wait(.05)
                    continue
                frame_id, received, frame = item
                previous = frame_id
                started = time.monotonic()
                crop, raw = preprocess(frame)
                with lock:
                    state['busy'] = True
                stdin, stdout, stderr = client.exec_command(inference_command(args), timeout=args.timeout)
                stdin.write(raw)
                stdin.flush()
                stdin.channel.shutdown_write()
                output = stdout.read()
                errors = stderr.read().decode(errors='replace')
                status = stdout.channel.recv_exit_status()
                stdout.channel.close()
                if status != 0 or len(output) != 4000:
                    raise RuntimeError(f'HPS runner failed ({status}): {errors}; output bytes={len(output)}')
                logits = np.frombuffer(output, dtype='<f4')
                if not np.isfinite(logits).all():
                    raise RuntimeError('Nonfinite HPS output')
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                top = np.argsort(-logits)[:5]
                lines = [f'{labels[i] if labels else "class " + str(i)}: {probabilities[i]:.1%}' for i in top]
                with lock:
                    state['result'] = (frame_id, received, crop, '\n'.join(lines), time.monotonic() - started)
                    state['busy'] = False
        except Exception as exc:
            if not stop.is_set():
                fail(str(exc))

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

    def refresh():
        with lock:
            latest, result, error, busy = (state[k] for k in ('latest', 'result', 'error', 'busy'))
        text = 'Waiting for camera and first prediction...'
        if latest:
            image = latest[2].copy()
            image.thumbnail((640, 480))
            live.image = ImageTk.PhotoImage(image)
            live.configure(image=live.image)
            text = f'Preview frame {latest[0]}; received {time.monotonic() - latest[1]:.1f}s ago. '
        if result:
            frame_id, received, crop, scores, elapsed = result
            classified.image = ImageTk.PhotoImage(crop)
            classified.configure(image=classified.image)
            text += (f'\nResult for frame {frame_id}; received {time.monotonic() - received:.1f}s ago. '
                     f'Request duration {elapsed:.2f}s (includes two inferences + load/transfer).\n{scores}')
        text += '\nClassifying newest available frame...' if busy else ''
        if error:
            text = 'STOPPED / STALE — restart after fixing:\n' + error + '\n' + text
        info.configure(text=text)
        root.after(100, refresh)

    def close():
        stop.set()
        client.close()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', close)
    if args.image:
        state['latest'] = (1, time.monotonic(), Image.open(args.image).convert('RGB'))
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
