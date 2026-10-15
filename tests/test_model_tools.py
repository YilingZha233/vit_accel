import csv
import json
import tempfile
import unittest
from pathlib import Path

from baseline.model_support import freeze_data, timing_stats, wilson


class ModelMetadataTests(unittest.TestCase):
    def test_timing_nearest_rank_and_group_independent(self):
        s = timing_stats(list(range(1, 21)))
        self.assertEqual(s['p95_ms'], 19)
        self.assertEqual(s['median_ms'], 10.5)
        with self.assertRaises(ValueError):
            timing_stats([float('nan')])

    def test_wilson_not_zero_width_for_perfect_sample(self):
        low, high = wilson(10, 10)
        self.assertLess(low, 100)
        self.assertAlmostEqual(high, 100)

    def test_manifest_rejects_cross_split_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'a.bin').write_bytes(b'same-image')
            (root/'b.bin').write_bytes(b'same-image')
            with (root/'data.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['id', 'path', 'label', 'split'])
                w.writerow(['a', 'a.bin', 0, 'calib'])
                w.writerow(['b', 'b.bin', 1, 'test'])
            with self.assertRaisesRegex(ValueError, 'Duplicate image'):
                freeze_data(root/'data.csv', root/'out.json')
            self.assertFalse((root/'out.json').exists())


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install model requirements for numerical tests')
class QuantizationTests(unittest.TestCase):
    def test_q3_q4_match_independent_integer_goldens(self):
        from baseline.quant_experiment import SimulatedLinear
        from baseline.reference import full_accumulation, rtl_compatibility
        torch.manual_seed(12)
        layer = torch.nn.Linear(32, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.randint(-127, 128, (2, 32)))
            layer.weight[0, 0] = 127  # fixes tensor weight scale to 1
            layer.bias.copy_(torch.tensor([.25, -.5]))
        x = torch.randint(-10, 11, (3, 32)).float()
        w = layer.weight.detach().int().tolist()
        xi = x.int().T.tolist()
        for variant, golden in [('Q3', full_accumulation), ('Q4', rtl_compatibility)]:
            sim = SimulatedLinear(layer, {'sx': 1., 'sy': 2.}, variant)
            expected = torch.tensor(golden(w, xi, 32768)).T.float()*2 + layer.bias
            torch.testing.assert_close(sim(x), expected, rtol=0, atol=0)

    def test_unrepresentable_scale_rejected(self):
        from baseline.quant_experiment import SimulatedLinear
        layer = torch.nn.Linear(16, 1, bias=False)
        with torch.no_grad():
            layer.weight.fill_(127)
        with self.assertRaisesRegex(ValueError, 'Q0.16'):
            SimulatedLinear(layer, {'sx': 2., 'sy': 1.}, 'Q3')

    def test_calibration_observes_bias_free_output(self):
        from baseline.quant_experiment import calibrate
        layer = torch.nn.Linear(16, 1)
        with torch.no_grad():
            layer.weight.fill_(1)
            layer.bias.fill_(100)
        model = torch.nn.Sequential(layer)
        c = calibrate(model, [torch.ones(1, 16)], '0')
        self.assertEqual(c['0']['output_absmax'], 16)

    def test_paired_loss_uses_corresponding_images(self):
        import numpy as np
        from scripts.compare_model_runs import compare_arrays
        ref, cand = np.zeros((4, 1000)), np.zeros((4, 1000))
        ref[:, 0] = 2
        cand[:, 0] = 2
        cand[0, 1] = 3
        result = compare_arrays(ref, cand, [0, 0, 0, 0], 1000)
        self.assertEqual(result['top1_loss_pp'], 25)
        self.assertEqual(result['prediction_agreement_pct'], 75)


if __name__ == '__main__':
    unittest.main()
