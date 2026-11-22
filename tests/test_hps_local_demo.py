import ast
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from urllib.request import urlopen
from urllib.error import HTTPError
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('local_demo',ROOT/'scripts/hps_local_demo.py')
app=importlib.util.module_from_spec(spec);spec.loader.exec_module(app)


class LocalDemoTests(unittest.TestCase):
    def test_python35_syntax(self):
        ast.parse((ROOT/'scripts/hps_local_demo.py').read_text(),feature_version=5)

    def test_preprocessing_reference(self):
        from baseline.model_support import transform
        ref=transform(ROOT/'configs/preprocessing.json')
        rng=np.random.default_rng(14)
        for w,h in [(640,480),(333,501),(501,333),(37,61)]:
            im=Image.fromarray(rng.integers(0,256,(h,w,3),dtype=np.uint8))
            _,raw=app.preprocess(im)
            np.testing.assert_array_equal(np.frombuffer(raw,dtype='<f4').reshape(3,224,224),ref(im).numpy())

    def test_jpeg_stream_fragments(self):
        parser=app.JPEGStream()
        frames=[]
        stream=b'junk\xff\xd8hello\xff\xd9\xff\xd8bye\xff\xd9'
        for b in stream:
            frames.extend(parser.feed(bytes([b])))
        self.assertEqual(frames,[b'\xff\xd8hello\xff\xd9',b'\xff\xd8bye\xff\xd9'])
        self.assertLess(len(parser.buffer),2)

    def test_snapshot_inference_without_browser_then_http(self):
        runner=ROOT/'build/model-runner-host/camera_runner'
        if not runner.exists():self.skipTest('Build host camera_runner')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'net.param').write_text('7767517\n3 3\nInput input 0 1 in0\nPooling avg 1 1 in0 pooled 0=1 4=1\nInnerProduct fc 1 1 pooled out0 0=1000 1=1 2=3000\n')
            (root/'net.bin').write_bytes(struct.pack('<I',0)+np.zeros(3000,dtype='<f4').tobytes()+np.arange(1000,dtype='<f4').tobytes())
            (root/'labels.txt').write_text('\n'.join('label'+str(i) for i in range(1000)))
            Image.new('RGB',(640,480),(128,30,80)).save(str(root/'image.jpg'))
            (root/'out').mkdir()
            args=SimpleNamespace(labels=str(root/'labels.txt'),runner=str(runner),param=str(root/'net.param'),
                model=str(root/'net.bin'),threads=2,precision='fp32',input_blob='in0',output_blob='out0',
                timeout=5,image=str(root/'image.jpg'),snapshot_out=str(root/'out'),timings=str(root/'times.csv'),max_frames=0)
            demo=app.Demo(args)
            try:
                demo.start()
                self.assertTrue(demo.stop.wait(10))
                demo.close()
                state=demo.state()
                self.assertEqual(state['error'],'')
                self.assertEqual(state['completed'],1)
                self.assertEqual(state['result']['top5'][0]['label'],'label999')
                self.assertEqual((root/'out/input.bin').stat().st_size,602112)
                self.assertEqual((root/'out/logits.bin').stat().st_size,4000)
                self.assertEqual(state['result']['first_request'],1)
                self.assertEqual(len((root/'times.csv').read_text().splitlines()),2)
                # Start web server only AFTER inference: no browser required to classify.
                server=app.Server(('127.0.0.1',0),app.handler_for(demo))
                worker=threading.Thread(target=server.serve_forever);worker.start()
                try:
                    base='http://127.0.0.1:'+str(server.server_address[1])
                    with urlopen(base+'/status',timeout=2) as r:
                        self.assertEqual(json.load(r)['result']['frame_id'],1)
                    with urlopen(base+'/preview.jpg',timeout=2) as r:
                        self.assertEqual(r.read(),(root/'image.jpg').read_bytes())
                    with self.assertRaises(HTTPError):urlopen(base+'/../../etc/passwd',timeout=2)
                finally:
                    server.shutdown();server.server_close();worker.join()
            finally:demo.close()


if __name__=='__main__':unittest.main()
