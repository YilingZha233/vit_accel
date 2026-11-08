import importlib.util
from pathlib import Path
import unittest
s = importlib.util.spec_from_file_location('kaggle_subset', Path(__file__).resolve().parents[1]/'scripts/prepare_kaggle_hps1000.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)

class InventoryTests(unittest.TestCase):
    def test_complete_mapping_uses_train_inventory_but_selects_only_val(self):
        wnids = ['n01440764','n01443537'] + [f'n02{i:06d}' for i in range(997)] + ['n15075141']
        names = [f'imagenet-mini/train/{w}/train.JPEG' for w in reversed(wnids)]
        names += [f'imagenet-mini/val/n15075141/ILSVRC2012_val_{i:08d}.JPEG' for i in range(1000)]
        mapping, val = m.inventory(names)
        self.assertEqual(mapping['n15075141'], 999)
        self.assertEqual(mapping['n01440764'], 0)
        self.assertEqual(len(val), 1000)
        self.assertTrue(all('/val/' in x for x in val))
    def test_partial_class_map_rejected(self):
        with self.assertRaises(ValueError):
            m.inventory(['imagenet-mini/val/n01440764/ILSVRC2012_val_00000001.JPEG'])

if __name__ == '__main__': unittest.main()
