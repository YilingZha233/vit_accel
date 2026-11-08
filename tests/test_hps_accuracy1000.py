import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import gzip


def module(name):
    p = Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py')
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


class AccuracyTests(unittest.TestCase):
    def test_hps_counts_and_platform_validation(self):
        m = module('summarize_hps_accuracy1000')
        ids = [str(i) for i in range(1000)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for b in range(10):
                d = root/'compact'/f'int8_batch_{b:02d}'; d.mkdir(parents=True)
                (d/'image_ids.json').write_text(json.dumps(ids[b*100:(b+1)*100]))
                r = dict(mode='eval', platform_machine='armv7l', precision_request='runtime-int8',
                         threads=2, completed=100, failed=0, accuracy_available=True,
                         top1_correct=80, top5_correct=95)
                (d/'run.json').write_text(json.dumps(r))
            result = m.aggregate(root,ids,'int8')
            self.assertEqual(result['n'], 1000)
            self.assertEqual(result['top1_pct'],80)
            self.assertEqual(result['top5_pct'],95)
            r['platform_machine']='x86_64'
            (d/'run.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError): m.aggregate(root,ids,'int8')

    def test_serial_decoder_rejects_missing_line(self):
        m = module('recover_hps_octal_archive')
        data = gzip.compress(bytes(range(256)))
        rows = [f'{i:07o} ' + ' '.join(f'{x:03o}' for x in data[i:i+16]) for i in range(0,len(data),16)]
        log = 'HPS_OCT_BEGIN\n' + '\n'.join(rows) + '\nHPS_OCT_END\n'
        self.assertEqual(m.decode(log), data)
        del rows[1]
        with self.assertRaises(ValueError):
            m.decode('HPS_OCT_BEGIN\n' + '\n'.join(rows) + '\nHPS_OCT_END\n')


if __name__ == '__main__': unittest.main()
