import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('subset', Path(__file__).resolve().parents[1] / 'scripts/prepare_imagenet_accuracy1000.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SubsetTests(unittest.TestCase):
    def test_disjoint_reproducible_sampling(self):
        a, b = m.select_indices(50000, 1000, 512, 4961000)
        self.assertEqual((len(a), len(b)), (1000, 512))
        self.assertFalse(set(a) & set(b))
        self.assertEqual((a, b), m.select_indices(50000, 1000, 512, 4961000))
        self.assertNotEqual(a, m.select_indices(50000, 1000, 512, 1)[0])

    def test_devkit_ids_are_not_class_indices(self):
        wnids = ['n01440764'] + [f'n02{i:06d}' for i in range(998)] + ['n15075141']
        synsets = [SimpleNamespace(ILSVRC2012_ID=1000-i, WNID=w, num_children=0)
                   for i, w in enumerate(wnids)]
        synsets.append(SimpleNamespace(ILSVRC2012_ID=1001, WNID='parent', num_children=2))
        labels, ordered = m.checkpoint_labels(synsets, [1000, 1, 999])
        self.assertEqual(labels, [0, 999, 1])
        self.assertEqual(ordered, wnids)

    def test_invalid_counts(self):
        with self.assertRaises(ValueError):
            m.select_indices(10, 10, 1, 0)


if __name__ == '__main__':
    unittest.main()
