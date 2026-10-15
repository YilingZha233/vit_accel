import unittest
from scripts.summarize_ncnn_profile import summarize


class NcnnProfileTests(unittest.TestCase):
    def fixture(self):
        # Runner executes first inference, one warmup, then two samples.
        text = '\n'.join('Gemm gemm_1 %.2fms | [32,49] -> [192,49]\nGELU gelu_1 0.00ms |'
                         % t for t in [99, 88, 2, 4])
        run = dict(mode='bench', warmups=1, completed=2, failed=0,
                   mean_ms=4., threads=2, precision_request='fp32')
        return text, run

    def test_excludes_first_and_warmup(self):
        text, run = self.fixture()
        layers, groups, summary = summarize(text, run, {**run, 'mean_ms': 3.5})
        self.assertEqual(layers[0]['mean_layer_ms'], 3)
        self.assertEqual(layers[0]['calls_per_image'], 1)
        self.assertEqual(summary['wall_minus_logged_layers_ms'], 1)
        self.assertAlmostEqual(summary['observed_profiling_overhead_pct'], 100/7)
        self.assertEqual(groups[1]['mean_layer_ms_per_image'], 0)

    def test_rejects_incomplete_log(self):
        text, run = self.fixture()
        with self.assertRaisesRegex(ValueError, 'expected'):
            summarize(text, {**run, 'completed': 3})

    def test_rejects_mismatched_control(self):
        text, run = self.fixture()
        with self.assertRaisesRegex(ValueError, 'threads'):
            summarize(text, run, {**run, 'threads': 1})

    def test_rejects_empty_profile(self):
        _, run = self.fixture()
        with self.assertRaisesRegex(ValueError, 'No layer'):
            summarize('ordinary console output', run)

    def repeated_fixture(self):
        lines = []
        for value in [100, 90, 80, 70, 1, 2, 3, 4, 5]:
            lines.extend(['Convolution conv_140 %.2fms |' % value,
                          'Split splitncnn_0 %.2fms |' % value,
                          'Split splitncnn_0 %.2fms |' % (2*value)])
        run = dict(mode='bench', warmups=3, completed=5, failed=0,
                   mean_ms=20., threads=2, precision_request='fp32')
        return '\n'.join(lines), run

    def test_repeated_split_18_records_in_9_passes(self):
        text, run = self.repeated_fixture()
        layers, groups, summary = summarize(text, run)
        split = next(r for r in layers if r['op_type'] == 'Split')
        self.assertEqual(split['calls_per_image'], 2)
        self.assertEqual(split['mean_layer_ms'], 9)
        self.assertEqual(split['median_layer_ms'], 9)
        self.assertEqual(split['mean_call_ms'], 4.5)
        self.assertEqual(summary['logged_layer_mean_sum_ms'], 12)
        self.assertEqual(summary['inference_passes'], 9)
        self.assertEqual(summary['skipped_inferences'], 4)
        self.assertEqual(summary['logged_calls_per_image'], 3)

    def test_rejects_different_measured_call_counts(self):
        text, run = self.repeated_fixture()
        # Remove just one Split call from the last measured inference.
        with self.assertRaisesRegex(ValueError, 'missing/extra'):
            summarize('\n'.join(text.splitlines()[:-1]), run)

    def test_warmup_call_counts_may_differ(self):
        text, run = self.repeated_fixture()
        # An initialization-only event must not pollute measured totals.
        lines = text.splitlines()
        lines.insert(1, 'Packing initialization_only 99.00ms |')
        layers, _, summary = summarize('\n'.join(lines), run)
        self.assertFalse(any(r['layer_name'] == 'initialization_only' for r in layers))
        self.assertEqual(summary['logged_layer_mean_sum_ms'], 12)


if __name__ == '__main__':
    unittest.main()
