"""Run with the model Python environment; host integration uses built ncnn runners."""
import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('live', ROOT / 'scripts/hps_camera_live.py')
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class Channel:
    closed = False
    def __init__(self, output):
        self.output, self.sent = bytearray(output), bytearray()
    def recv_ready(self): return bool(self.output)
    def recv_stderr_ready(self): return False
    def exit_status_ready(self): return not self.output
    def send_ready(self): return True
    def recv(self, n):
        n = min(n, 7)  # Deliberately fragment every protocol field.
        out = bytes(self.output[:n]); del self.output[:n]; return out
    def send(self, value):
        n = min(len(value), 8191)
        self.sent.extend(value[:n]); return n


class ProtocolTests(unittest.TestCase):
    def test_fragmented_protocol(self):
        logits = np.arange(1000, dtype='<f4')
        ch = Channel(struct.pack('<4sd', b'RDY1', 10.) +
                     struct.pack('<4sIddd', b'RES1', 17, .1, 5., 5.2) + logits.tobytes())
        session = live.PersistentSession(ch, 2)
        session.ready()
        y, pack, infer, service = session.predict(17, bytes(602112))
        np.testing.assert_array_equal(y, logits)
        self.assertEqual(session.load_ms, 10)
        self.assertEqual(infer, 5)
        self.assertEqual(ch.sent, struct.pack('<4sI', b'VIT1', 17) + bytes(602112))

    def test_wrong_frame_rejected(self):
        ch = Channel(struct.pack('<4sIddd', b'RES1', 99, 0, 1, 1) + bytes(4000))
        with self.assertRaisesRegex(RuntimeError, 'Mismatched'):
            live.PersistentSession(ch, 1).predict(1, bytes(602112))

    def test_closed_and_stalled_streams(self):
        with self.assertRaisesRegex(RuntimeError, 'stopped'):
            live.PersistentSession(Channel(b''), 1).ready()
        ch = Channel(b'')
        ch.exit_status_ready = lambda: False
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            live.PersistentSession(ch, .02).ready()
        self.assertLess(time.monotonic()-started, .5)

    def test_nonfinite_logits_rejected(self):
        y = np.zeros(1000, dtype='<f4'); y[0] = np.nan
        ch = Channel(struct.pack('<4sIddd', b'RES1', 1, 0, 1, 1) + y.tobytes())
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            live.PersistentSession(ch, 1).predict(1, bytes(602112))

    def test_reference_preprocessing(self):
        from baseline.model_support import transform
        reference = transform(ROOT / 'configs/preprocessing.json')
        rng = np.random.default_rng(8)
        for w,h in [(640,480),(480,640),(333,501),(501,333),(1920,1080),(37,61)]:
            im = Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
            _,raw = live.preprocess(im)
            np.testing.assert_array_equal(np.frombuffer(raw,dtype='<f4').reshape(3,224,224), reference(im).numpy())


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = Path(os.environ.get('CAMERA_RUNNER', ROOT / 'build/model-runner-host/camera_runner'))
        cls.baseline = Path(os.environ.get('MODEL_RUNNER', ROOT / 'build/model-runner-host/model_runner'))
        if not cls.runner.exists() or not cls.baseline.exists():
            raise unittest.SkipTest('Build host camera_runner and model_runner first')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.param = self.root / 'test.param'
        self.model = self.root / 'test.bin'
        self.param.write_text('7767517\n3 3\nInput input 0 1 in0\nPooling avg 1 1 in0 pooled 0=1 4=1\nInnerProduct fc 1 1 pooled out0 0=1000 1=1 2=3000\n')
        rng = np.random.default_rng(42)
        self.model.write_bytes(struct.pack('<I',0) + rng.normal(size=3000).astype('<f4').tobytes() + rng.normal(size=1000).astype('<f4').tobytes())
        self.args = [str(self.runner),str(self.param),str(self.model),'2','in0','out0','fp32']

    def test_repeated_requests_match_benchmark(self):
        rng = np.random.default_rng(3)
        inputs = [rng.normal(size=3*224*224).astype('<f4').tobytes() for _ in range(3)]
        data = b''.join(struct.pack('<4sI',b'VIT1',i+10)+raw for i,raw in enumerate(inputs))
        result = subprocess.run(self.args,input=data,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr.decode())
        self.assertEqual(len(result.stdout),12+4032*3)
        self.assertEqual(result.stdout[:4],b'RDY1')
        for i,raw in enumerate(inputs):
            offset=12+i*4032
            magic,frame,pack,infer,service=struct.unpack('<4sIddd',result.stdout[offset:offset+32])
            self.assertEqual((magic,frame),(b'RES1',i+10))
            self.assertGreaterEqual(service,infer)
            self.assertGreaterEqual(pack,0)
            actual=np.frombuffer(result.stdout[offset+32:offset+4032],dtype='<f4')
            (self.root/'input.bin').write_bytes(raw)
            (self.root/'input.tsv').write_text('frame\t0\tinput.bin\n')
            out=self.root/f'out{i}'
            cmd=[str(self.baseline),'eval',str(self.param),str(self.model),str(self.root/'input.tsv'),str(out),'2','0','1','in0','out0','0','fp32']
            base=subprocess.run(cmd,capture_output=True,timeout=20)
            self.assertEqual(base.returncode,0,base.stderr.decode())
            expected=np.fromfile(out/'00000.logits.bin',dtype='<f4')
            np.testing.assert_array_equal(actual,expected)

    def test_invalid_inputs_and_load_failure(self):
        bad = np.zeros(3*224*224,dtype='<f4'); bad[0]=np.nan
        for data in [b'bad!', b'VIT1', struct.pack('<4sI',b'VIT1',1)+bad.tobytes()]:
            result=subprocess.run(self.args,input=data,capture_output=True,timeout=10)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(len(result.stdout),12)  # Ready only; no fake result.
        args=self.args.copy(); args[1]=str(self.root/'missing.param')
        result=subprocess.run(args,input=b'',capture_output=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(result.stdout,b'')

if __name__ == '__main__': unittest.main()
