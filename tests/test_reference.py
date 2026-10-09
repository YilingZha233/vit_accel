import random
import unittest
from baseline.reference import (full_accumulation, rtl_compatibility, hardware_shape,
    pack_inputs, pack_output, unpack_output, weight_offset, input_offset, output_offset)


class ReferenceTests(unittest.TestCase):
    def test_known_matmul(self):
        # 2x2 mathematical example, inputs doubled to cancel scale 0.5.
        self.assertEqual(full_accumulation([[2,4],[6,8]], [[5,6],[7,8]], 32768), [[19,22],[43,50]])

    def test_identity_all_int8_values(self):
        w = [[2 if r == p else 0 for p in range(16)] for r in range(16)]
        x = [[p * 16 + c - 128 for c in range(16)] for p in range(16)]
        self.assertEqual(full_accumulation(w,x,32768), x)
        self.assertEqual(rtl_compatibility(w,x,32768), x)

    def test_negative_shift_floors(self):
        self.assertEqual(full_accumulation([[-1]], [[1]], 32768), [[-1]])

    def test_rounding_gap(self):
        w = [[1 if p in (0,16) else 0 for p in range(32)]]
        x = [[1] for _ in range(32)]
        self.assertEqual(full_accumulation(w,x,32768), [[1]])
        self.assertEqual(rtl_compatibility(w,x,32768), [[0]])

    def test_saturation_cancellation(self):
        w, x = [[127]*16 + [-127]*16], [[127] for _ in range(32)]
        self.assertEqual(full_accumulation(w,x,32768), [[0]])
        self.assertEqual(rtl_compatibility(w,x,32768), [[-1]])

    def test_scale_bounds(self):
        self.assertEqual(full_accumulation([[127]],[[127]],0), [[0]])
        self.assertEqual(full_accumulation([[127]],[[127]],65535), [[127]])
        for q in (-1,65536):
            with self.assertRaises(ValueError):
                full_accumulation([[1]],[[1]],q)

    def test_invalid_shapes_and_values(self):
        for w,x in [([],[[1]]), ([[1,2]],[[1]]), ([[128]],[[1]])]:
            with self.assertRaises(ValueError):
                full_accumulation(w,x,1)
        for dims in [(16,17,1),(17,1024,1),(16,32,129),(32,16,65),(0,16,1)]:
            with self.assertRaises(ValueError):
                hardware_shape(*dims)

    def test_banked_addresses(self):
        self.assertEqual(weight_offset(1,0,32),1024)
        self.assertEqual(weight_offset(16,3,32),35)
        self.assertEqual(input_offset(17,2,18),276)
        self.assertEqual(output_offset(16,2,18),20)
        w = [[r-16]*32 for r in range(25)]
        x = [[p-16]*18 for p in range(32)]
        wi,xi = pack_inputs(w,x)
        self.assertEqual(wi[1024], (-15)&255)
        self.assertEqual(wi[32], 0)
        self.assertEqual(xi[276], 1)

    def test_output_layout(self):
        y = [[r-20+c for c in range(18)] for r in range(25)]
        image = pack_output(y)
        self.assertEqual(image[128], (-19)&255)
        self.assertEqual(image[18], (-4)&255)
        self.assertEqual(unpack_output(image,25,18),y)

    def test_seeded_1000_single_block_cases(self):
        rng = random.Random(496)
        for _ in range(1000):
            k,m = rng.randint(1,4),rng.randint(1,4)
            w = [[rng.randint(-128,127) for _ in range(16)] for _ in range(k)]
            x = [[rng.randint(-128,127) for _ in range(m)] for _ in range(16)]
            q = rng.choice([0,1,255,32768,65535])
            self.assertEqual(full_accumulation(w,x,q),rtl_compatibility(w,x,q))


if __name__ == "__main__":
    unittest.main()
