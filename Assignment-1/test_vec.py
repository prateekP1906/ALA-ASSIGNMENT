import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from vec import Vec as Vector
from w2v import get_word_vector, load_model


class TestVector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = load_model("glove50/glove_50_fast.wordvectors")
        raw_vec = get_word_vector(cls.model, "india")
        cls.v = Vector([float(x) for x in raw_vec])
        cls.raw = [float(x) for x in raw_vec]

    # --- Mean Tests ---
    def test_mean_word(self):
        expected_mean = sum(self.raw) / len(self.raw)
        actual_mean = self.v.mean()
        print(f"[test_mean_word] Expected: {expected_mean:.6f} | Computed: {actual_mean:.6f}")
        self.assertAlmostEqual(actual_mean, expected_mean, places=5)

    def test_mean_definition(self):
        """Property: Mean is sum(x) / len(x)."""
        expected_mean = sum(self.raw) / len(self.raw)
        actual_mean = self.v.mean()
        print(f"[test_mean_definition] Expected: {expected_mean:.6f} | Computed: {actual_mean:.6f}")
        self.assertAlmostEqual(actual_mean, expected_mean, places=5)

    def test_mean_constant_vector(self):
        """Property: The mean of a constant vector [c, c, ..., c] is c."""
        c = 4.5
        v_const = Vector([c] * 10)
        actual_mean = v_const.mean()
        print(f"[test_mean_constant_vector] Constant: {c} | Vector Mean: {actual_mean}")
        self.assertAlmostEqual(actual_mean, c, places=5)

    # --- Demean Tests ---
    def test_demean_returns_vector(self):
        """Property: Demean returns a Vec instance."""
        demeaned_v = self.v.demean()
        print(f"[test_demean_returns_vector] Return type: {type(demeaned_v).__name__}")
        self.assertIsInstance(demeaned_v, Vector)

    def test_demean_mean_is_zero(self):
        """Property: The mean of a de-meaned vector is always 0."""
        demeaned_mean = self.v.demean().mean()
        print(f"[test_demean_mean_is_zero] Mean of demeaned vector: {demeaned_mean:.6e}")
        self.assertAlmostEqual(demeaned_mean, 0.0, places=5)

    def test_demean_sum_is_zero(self):
        """Property: The sum of entries of a de-meaned vector is 0."""
        demeaned_v = self.v.demean()
        total_sum = sum(demeaned_v)
        print(f"[test_demean_sum_is_zero] Sum of demeaned vector entries: {total_sum:.6e}")
        self.assertAlmostEqual(total_sum, 0.0, places=5)

    def test_demean_preserves_length(self):
        """Property: A de-meaned vector has the same dimension as original."""
        orig_len = len(self.v)
        demeaned_len = len(self.v.demean())
        print(f"[test_demean_preserves_length] Original len: {orig_len} | Demeaned len: {demeaned_len}")
        self.assertEqual(demeaned_len, orig_len)

    # --- Standard Deviation Tests ---
    def test_standard_deviation_word(self):
        expected_std_dev = (sum((x - self.v.mean()) ** 2 for x in self.raw) / len(self.raw)) ** 0.5
        actual_std = self.v.std()
        print(f"[test_standard_deviation_word] Expected Std: {expected_std_dev:.6f} | Computed Std: {actual_std:.6f}")
        self.assertAlmostEqual(actual_std, expected_std_dev, places=5)

    def test_std_non_negativity(self):
        """Property: Standard deviation is always non-negative (>= 0)."""
        actual_std = self.v.std()
        print(f"[test_std_non_negativity] Computed Std (>= 0): {actual_std:.6f}")
        self.assertGreaterEqual(actual_std, 0.0)

    def test_std_constant_vector(self):
        """Property: The standard deviation of a constant vector is exactly 0."""
        v_const = Vector([3.0] * 10)
        actual_std = v_const.std()
        print(f"[test_std_constant_vector] Constant Vector Std (expected 0.0): {actual_std}")
        self.assertAlmostEqual(actual_std, 0.0, places=5)

    def test_std_matches_demeaned_norm(self):
        """Property: std is sqrt(mean of squares of demeaned vector)."""
        demeaned_v = self.v.demean()
        expected_std = (sum(x ** 2 for x in demeaned_v) / len(demeaned_v)) ** 0.5
        actual_std = self.v.std()
        print(f"[test_std_matches_demeaned_norm] Expected: {expected_std:.6f} | Computed: {actual_std:.6f}")
        self.assertAlmostEqual(actual_std, expected_std, places=5)


if __name__ == "__main__":
    unittest.main()