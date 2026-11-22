"""Local contract checks; these do not substitute for board/camera testing."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('camera_viewer', ROOT / 'scripts/hps_camera_viewer.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class CameraContractTests(unittest.TestCase):
    def test_preprocessing_matches_existing_reference(self):
        from baseline.model_support import transform
        reference = transform(ROOT / 'configs/preprocessing.json')
        rng = np.random.default_rng(2026)
        for width, height in [(640, 480), (480, 640), (1920, 1080), (333, 501),
                              (501, 333), (224, 224), (37, 61), (256, 257)]:
            with self.subTest(size=(width, height)):
                image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
                crop, payload = viewer.preprocess(image)
                self.assertEqual(crop.size, (224, 224))
                self.assertEqual(len(payload), 602112)
                actual = np.frombuffer(payload, dtype='<f4').reshape(3, 224, 224)
                np.testing.assert_array_equal(actual, reference(image).numpy())

    def test_remote_command_contract_and_quoting(self):
        with tempfile.TemporaryDirectory(prefix='camera test ') as folder:
            root = Path(folder)
            runner = root / "runner's test"
            runner.write_text('''#!/usr/bin/env python3
import pathlib, sys
assert sys.argv[1] == 'eval'
assert sys.argv[6:] == ['2', '0', '1', 'in0', 'out0', '0', 'runtime-int8']
tsv = pathlib.Path(sys.argv[4])
assert tsv.read_text() == 'camera\\t0\\tinput.bin\\n'
assert (tsv.parent / 'input.bin').read_bytes() == b'x' * 602112
out = pathlib.Path(sys.argv[5])
assert not out.exists()
out.mkdir()
(out / '00000.logits.bin').write_bytes(b'1234' * 1000)
print('runner diagnostic')
''')
            runner.chmod(0o700)
            args = SimpleNamespace(runner=str(runner), param="model's file.param", model='model file.bin',
                                   threads=2, input_blob='in0', output_blob='out0', precision='runtime-int8')
            result = subprocess.run(['sh', '-c', viewer.inference_command(args)], input=b'x' * 602112,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(result.stdout, b'1234' * 1000)
            self.assertIn(b'runner diagnostic', result.stderr)

    def test_remote_command_rejects_short_tensor(self):
        args = SimpleNamespace(runner='/not/run', param='model.param', model='model.bin',
                               threads=2, input_blob='in0', output_blob='out0', precision='runtime-int8')
        result = subprocess.run(['sh', '-c', viewer.inference_command(args)], input=b'bad',
                                capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b'')


if __name__ == '__main__':
    unittest.main()
