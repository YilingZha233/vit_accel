#!/usr/bin/env python3
"""Run on HPS: capture, decode, preprocess, ncnn inference and labels.
Python 3.5+ compatible. Browser is display-only; inference never depends on it.
"""
import argparse
import base64
import csv
import io
import json
import os
import select
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
import numpy as np
from PIL import Image
import PIL


def preprocess(image):
    image = image.convert('RGB')
    w, h = image.size
    size = (256, int(256*h/w)) if w <= h else (int(256*w/h), 256)
    bicubic = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC
    image = image.resize(size, bicubic)
    left, top = round((size[0]-224)/2), round((size[1]-224)/2)
    crop = image.crop((left, top, left+224, top+224))
    x = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)/np.float32(255)
    x = x - np.array([.485, .456, .406], dtype=np.float32)[:, None, None]
    x /= np.array([.229, .224, .225], dtype=np.float32)[:, None, None]
    return crop, x.astype('<f4').tobytes()


def verify_fixtures(folder):
    """No reference preprocessing on Windows is needed at deployment time."""
    folder = Path(folder)
    cases = json.loads((folder/'manifest.json').read_text())
    passed = True
    for case in cases:
        image = Image.open(str(folder/case['image'])).convert('RGB')
        _, raw = preprocess(image)
        actual = np.frombuffer(raw, dtype='<f4')
        expected = np.fromfile(str(folder/case['tensor']), dtype='<f4')
        if actual.shape != expected.shape:
            raise ValueError('Invalid fixture shape')
        error = float(np.max(np.abs(actual-expected)))
        ok = bool(np.isfinite(actual).all() and error <= 1e-5)
        print('{} max_abs_error={:.9g} {}'.format(case['image'], error, 'PASS' if ok else 'FAIL'))
        passed = passed and ok
    print('Pillow {}; NumPy {}'.format(PIL.__version__, np.__version__))
    return passed


class Process:
    def __init__(self, args):
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=0)
        self.errors = deque(maxlen=8)
        self.error_lock = threading.Lock()
        def drain():
            while True:
                data = self.proc.stderr.read(1024)
                if not data:
                    break
                with self.error_lock:
                    self.errors.append(data)
        self.reader = threading.Thread(target=drain)
        self.reader.daemon = True
        self.reader.start()

    def diagnostic(self):
        with self.error_lock:
            return b''.join(self.errors).decode('utf-8', errors='replace')

    def read_exact(self, size, deadline):
        data = bytearray()
        fd = self.proc.stdout.fileno()
        while len(data) < size:
            left = deadline-time.monotonic()
            if left <= 0:
                raise RuntimeError('Runner timeout. '+self.diagnostic())
            ready, _, _ = select.select([fd], [], [], min(left, .1))
            if ready:
                chunk = os.read(fd, size-len(data))
                if not chunk:
                    raise RuntimeError('Runner closed output. '+self.diagnostic())
                data.extend(chunk)
        return bytes(data)

    def write_exact(self, data, deadline):
        fd = self.proc.stdin.fileno()
        offset = 0
        while offset < len(data):
            left = deadline-time.monotonic()
            if left <= 0:
                raise RuntimeError('Runner input timeout')
            _, ready, _ = select.select([], [fd], [], min(left, .1))
            if ready:
                # <= PIPE_BUF on Linux: a writable pipe can accept this bounded write.
                n = os.write(fd, data[offset:offset+4096])
                if n <= 0:
                    raise RuntimeError('Runner input closed')
                offset += n

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.reader.join(timeout=1)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            stream.close()


class Backend:
    """Replace this backend later with the team's hybrid full-model backend."""
    def __init__(self, args):
        self.timeout = args.timeout
        self.process = Process([args.runner, args.param, args.model, str(args.threads),
                                args.input_blob, args.output_blob, args.precision])
        try:
            ready = self.process.read_exact(12, time.monotonic()+self.timeout)
            magic, self.load_ms = struct.unpack('<4sd', ready)
            if magic != b'RDY1' or not np.isfinite(self.load_ms) or self.load_ms < 0:
                raise RuntimeError('Invalid runner ready message')
        except Exception:
            self.process.close()
            raise

    def predict(self, frame_id, raw):
        if len(raw) != 602112:
            raise ValueError('Invalid tensor length')
        deadline = time.monotonic()+self.timeout
        self.process.write_exact(struct.pack('<4sI', b'VIT1', frame_id)+raw, deadline)
        header = self.process.read_exact(32, deadline)
        magic, returned, pack, infer, service = struct.unpack('<4sIddd', header)
        if magic != b'RES1' or returned != frame_id:
            raise RuntimeError('Wrong runner response/frame ID')
        if not all(np.isfinite(t) and t >= 0 for t in (pack, infer, service)):
            raise RuntimeError('Invalid timing fields')
        y = np.frombuffer(self.process.read_exact(4000, deadline), dtype='<f4')
        if not np.isfinite(y).all():
            raise RuntimeError('Nonfinite logits')
        return y, pack, infer, service

    def close(self):
        self.process.close()


class JPEGStream:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > 8*1024*1024:
            raise RuntimeError('Oversized/invalid MJPEG stream')
        frames = []
        while True:
            start = self.buffer.find(b'\xff\xd8')
            if start < 0:
                self.buffer[:] = self.buffer[-1:]
                break
            end = self.buffer.find(b'\xff\xd9', start+2)
            if end < 0:
                del self.buffer[:start]
                break
            frames.append(bytes(self.buffer[start:end+2]))
            del self.buffer[:end+2]
        return frames


PAGE = b'''<!doctype html><html><head><meta charset="utf-8"><title>HPS TinyViT</title>
<style>body{font:18px system-ui;background:#111827;color:#eef2ff;margin:24px}section{display:flex;gap:24px;flex-wrap:wrap}img{max-width:100%;background:#263244}#live{width:640px}#crop{width:224px}pre{white-space:pre-wrap}small{color:#cbd5e1}</style></head>
<body><h1>DE1-SoC: HPS-only TinyViT</h1><p>Capture, preprocessing, inference and labels run on HPS. This browser only displays results.</p>
<section><div><h3>Live preview</h3><img id="live" alt="Waiting for camera"></div><div><h3>Last classified crop</h3><img id="crop" alt="Waiting for result"><pre id="labels"></pre></div></section>
<pre id="status">Connecting...</pre><small>Scores are softmax, not calibrated accuracy. Preview rate differs from classification rate.</small>
<script>
let shown=-1, lastFrame=-1;
async function poll(){try{const r=await fetch('/status',{cache:'no-store',signal:AbortSignal.timeout(5000)});if(!r.ok)throw Error('HTTP '+r.status);const s=await r.json();
if(s.frame_id!==lastFrame){document.getElementById('live').src='/preview.jpg?v='+s.frame_id;lastFrame=s.frame_id;}
let text=s.status+' | completed: '+s.completed+' | preview received age: '+(s.preview_age_s===null?'n/a':s.preview_age_s.toFixed(1)+' s');
if(s.result){let q=s.result;if(q.frame_id!==shown){document.getElementById('crop').src='data:image/jpeg;base64,'+q.crop_b64;document.getElementById('labels').textContent=q.top5.map(x=>x.label+': '+(100*x.score).toFixed(1)+'%').join('\\n');shown=q.frame_id;}
text+='\\nResult frame '+q.frame_id+' | frame age '+s.result_age_s.toFixed(1)+' s';text+='\\nHPS JPEG decode '+q.decode_ms.toFixed(1)+' ms | preprocessing '+q.preprocess_ms.toFixed(1)+' ms';text+='\\nHPS inference '+q.infer_ms.toFixed(0)+' ms | request '+(q.request_ms/1000).toFixed(2)+' s';}
if(s.error)text+='\\nSTOPPED / STALE: '+s.error;document.getElementById('status').textContent=text;
}catch(e){document.getElementById('status').textContent='DISCONNECTED / STALE: '+e.message;}setTimeout(poll,200);}poll();
</script></body></html>'''


FIELDS = ['frame_id','first_request','decode_ms','preprocess_ms','pack_ms','infer_ms',
          'service_ms','ipc_overhead_ms','request_ms','received_to_result_ms',
          'publish_ms','completed_since_start_s','load_ms','top1']


class Demo:
    def __init__(self, args):
        self.args = args
        self.labels = Path(args.labels).read_text(encoding='utf-8').splitlines()
        if len(self.labels) != 1000:
            raise ValueError('Labels must contain 1000 ordered lines')
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.latest = None
        self.result = None
        self.result_received = None
        self.status, self.error = 'STARTING', ''
        self.completed = 0
        self.started = time.monotonic()
        self.workers = []

    def fail(self, exc):
        with self.lock:
            self.error, self.status = str(exc), 'STOPPED'
        self.stop.set()
        print('STOPPED: '+str(exc), flush=True)

    def capture(self):
        proc = None
        try:
            if self.args.image:
                jpeg = Path(self.args.image).read_bytes()
                Image.open(io.BytesIO(jpeg)).verify()
                with self.lock:
                    self.latest = (1, time.monotonic(), jpeg)
                return
            proc = Process([self.args.ffmpeg, '-nostdin','-hide_banner','-loglevel','error',
                            '-f','v4l2','-input_format','mjpeg','-video_size',self.args.size,
                            '-framerate',str(self.args.fps),'-i',self.args.device,
                            '-an','-c:v','copy','-f','image2pipe','pipe:1'])
            parser, last, frame_id = JPEGStream(), time.monotonic(), 0
            fd = proc.proc.stdout.fileno()
            while not self.stop.is_set():
                ready, _, _ = select.select([fd],[],[],.2)
                if not ready:
                    if time.monotonic()-last > 15:
                        raise RuntimeError('Camera timeout. '+proc.diagnostic())
                    continue
                data = os.read(fd,65536)
                if not data:
                    raise RuntimeError('Camera stopped. '+proc.diagnostic())
                for jpeg in parser.feed(data):
                    frame_id += 1
                    last = time.monotonic()
                    with self.lock:
                        self.latest = (frame_id,last,jpeg)
        except Exception as exc:
            if not self.stop.is_set():
                self.fail(exc)
        finally:
            if proc:
                proc.close()

    def classify(self):
        backend, log = None, None
        try:
            backend = Backend(self.args)
            if self.args.timings:
                log = open(self.args.timings,'x',newline='')
                writer = csv.DictWriter(log,fieldnames=FIELDS)
                writer.writeheader()
            previous = -1
            while not self.stop.is_set():
                with self.lock:
                    item = self.latest
                if not item or item[0] == previous:
                    self.stop.wait(.02)
                    continue
                frame_id, received, jpeg = item
                previous = frame_id
                with self.lock:
                    self.status = 'CLASSIFYING'
                start = time.monotonic()
                image = Image.open(io.BytesIO(jpeg)).convert('RGB')
                decoded = time.monotonic()
                crop, raw = preprocess(image)
                prepared = time.monotonic()
                y, pack, infer, service = backend.predict(frame_id,raw)
                returned = time.monotonic()
                scores = np.exp(y-y.max()); scores /= scores.sum()
                indices = np.argsort(-y)[:5]
                top5 = [dict(index=int(i),label=self.labels[int(i)],score=float(scores[i])) for i in indices]
                crop_jpeg = io.BytesIO(); crop.save(crop_jpeg,format='JPEG',quality=90)
                crop_bytes = crop_jpeg.getvalue()
                m = dict(frame_id=frame_id,first_request=int(self.completed==0),
                         decode_ms=(decoded-start)*1000,preprocess_ms=(prepared-decoded)*1000,
                         pack_ms=pack,infer_ms=infer,service_ms=service,
                         ipc_overhead_ms=max(0,(returned-prepared)*1000-service),
                         request_ms=(returned-start)*1000,received_to_result_ms=(returned-received)*1000,
                         load_ms=backend.load_ms,top1=int(indices[0]))
                m['publish_ms'] = (time.monotonic()-returned)*1000
                m['completed_since_start_s'] = time.monotonic()-self.started
                result = dict(m,top5=top5,crop_b64=base64.b64encode(crop_bytes).decode('ascii'),
                              backend='hps-cpu',precision=self.args.precision)
                if self.args.snapshot_out:
                    out = Path(self.args.snapshot_out)
                    (out/'input.bin').write_bytes(raw)
                    (out/'logits.bin').write_bytes(y.astype('<f4').tobytes())
                    (out/'crop.jpg').write_bytes(crop_bytes)
                    (out/'result.json').write_text(json.dumps(result,indent=2))
                if log:
                    writer.writerow(m); log.flush()
                with self.lock:
                    self.completed += 1
                    self.result, self.result_received = result, received
                    self.status = 'READY'
                print('frame={} class={} inference_ms={:.1f} preprocess_ms={:.1f} request_ms={:.1f}'.format(
                    frame_id,self.labels[int(indices[0])],infer,m['preprocess_ms'],m['request_ms']),flush=True)
                if self.args.image or (self.args.max_frames and self.completed >= self.args.max_frames):
                    with self.lock:
                        self.status = 'COMPLETE (retained snapshot)'
                    self.stop.set()
        except Exception as exc:
            if not self.stop.is_set():
                self.fail(exc)
        finally:
            if log:
                log.close()
            if backend:
                backend.close()

    def state(self):
        now = time.monotonic()
        with self.lock:
            return dict(status=self.status,error=self.error,completed=self.completed,
                        frame_id=self.latest[0] if self.latest else None,
                        preview_age_s=now-self.latest[1] if self.latest else None,
                        result_age_s=now-self.result_received if self.result_received is not None else None,
                        result=self.result)

    def start(self):
        for target in (self.capture,self.classify):
            worker = threading.Thread(target=target)
            worker.daemon = True
            self.workers.append(worker); worker.start()

    def close(self):
        self.stop.set()
        # In-flight inference completes or hits its bounded timeout before cleanup.
        for worker in self.workers:
            worker.join(timeout=self.args.timeout+5)


def handler_for(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            path = self.path.split('?',1)[0]
            if path == '/':
                data, mime = PAGE, 'text/html; charset=utf-8'
            elif path == '/status':
                data, mime = json.dumps(demo.state()).encode('utf-8'), 'application/json'
            elif path == '/preview.jpg':
                with demo.lock:
                    data = demo.latest[2] if demo.latest else None
                if data is None:
                    self.send_error(503,'Waiting for camera'); return
                mime = 'image/jpeg'
            else:
                self.send_error(404); return
            try:
                self.connection.settimeout(2)
                self.send_response(200)
                self.send_header('Content-Type',mime)
                self.send_header('Content-Length',str(len(data)))
                self.send_header('Cache-Control','no-store')
                self.end_headers(); self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError,socket.timeout): pass
    return Handler


class Server(ThreadingMixIn,HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-fixtures')
    p.add_argument('--runner'); p.add_argument('--param'); p.add_argument('--model'); p.add_argument('--labels')
    p.add_argument('--threads',type=int,choices=[1,2],default=2)
    p.add_argument('--precision',choices=['fp32','runtime-int8'],default='runtime-int8')
    p.add_argument('--input-blob',default='in0'); p.add_argument('--output-blob',default='out0')
    p.add_argument('--ffmpeg',default='ffmpeg'); p.add_argument('--device',default='/dev/video0')
    p.add_argument('--size',default='640x480'); p.add_argument('--fps',type=int,default=30)
    p.add_argument('--image'); p.add_argument('--snapshot-out'); p.add_argument('--timings')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--bind',default='0.0.0.0'); p.add_argument('--port',type=int,default=8080)
    p.add_argument('--timeout',type=float,default=120)
    a=p.parse_args()
    if a.verify_fixtures:
        raise SystemExit(0 if verify_fixtures(a.verify_fixtures) else 1)
    if not all((a.runner,a.param,a.model,a.labels)):
        p.error('--runner, --param, --model and --labels are required')
    if a.fps <= 0 or a.timeout <= 0 or a.max_frames < 0:
        p.error('Invalid rate/timeout/count')
    if a.timings and Path(a.timings).exists(): p.error('Timings file exists; choose a new filename')
    if a.snapshot_out:
        if not a.image: p.error('--snapshot-out requires --image')
        Path(a.snapshot_out).mkdir()  # New directory only; parent must exist.
    demo=Demo(a)
    server=Server((a.bind,a.port),handler_for(demo))
    print('HPS local demo: browser http://BOARD_IP:{}/ ; Ctrl+C to stop'.format(a.port),flush=True)
    print('Pillow {}; NumPy {}'.format(PIL.__version__,np.__version__),flush=True)
    demo.start()
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        print('Stopping...',flush=True)
    finally:
        server.server_close(); demo.close()


if __name__ == '__main__': main()
