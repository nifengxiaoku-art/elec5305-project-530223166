"""Scientific invariants, including known source-overlap regression cases."""
import unittest
import numpy as np
from src.dataset import load_dataset, split_rows, sample_rows, CATEGORIES
from src.evaluate import metrics
from src.augmentation import add_noise
from src.few_shot import fit_classifier


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full, cls.rows = load_dataset("data/ESC-50")

    def test_counts_and_splits(self):
        self.assertEqual(len(self.full), 2000)
        self.assertEqual(len(self.rows), 400)
        self.assertTrue((self.rows.groupby(["category", "fold"]).size() == 8).all())
        for fold in range(1, 6):
            train, val, test = split_rows(self.rows, fold)
            self.assertEqual(len(test), 80)
            for a, b in [(train, val), (train, test), (val, test)]:
                self.assertFalse(set(a.src_file) & set(b.src_file))
                self.assertFalse(set(a.filename) & set(b.filename))
            self.assertTrue((sample_rows(train, 16, 17).groupby("category").size() == 16).all())

    def test_nested_sampling_and_infeasible_24(self):
        for fold in range(1, 6):
            train, _, _ = split_rows(self.rows, fold)
            a, b = sample_rows(train, 4, 17), sample_rows(train, 16, 17)
            self.assertTrue(set(a.filename).issubset(b.filename))
            self.assertEqual(a.filename.tolist(), sample_rows(train, 4, 17).filename.tolist())
            if fold in [1, 2, 4]:
                with self.assertRaises(ValueError):
                    sample_rows(train, 24, 17)
            else:
                self.assertEqual(len(sample_rows(train, 24, 17)), 240)

    def test_noise_snr_determinism_and_silence(self):
        x = np.sin(np.arange(44100) * .1).astype(np.float32)
        noisy, details = add_noise(x, 20, 8)
        self.assertAlmostEqual(details["measured_snr_db"], 20, places=4)
        np.testing.assert_array_equal(noisy, add_noise(x, 20, 8)[0])
        with self.assertRaises(ValueError):
            add_noise(np.zeros(100), 20, 8)

    def test_metrics_known_errors(self):
        y = np.repeat(CATEGORIES, 8)
        predicted = y.copy()
        predicted[np.where(y == "hen")[0][:6]] = "rooster"
        predicted[np.where(y == "pig")[0][:1]] = "rooster"
        result = metrics(y, predicted)
        self.assertAlmostEqual(result["accuracy"], .9125)
        self.assertAlmostEqual(result["macro_f1"], .9028985507246376)
        self.assertEqual(np.asarray(result["confusion_matrix"]).sum(), 80)

    def test_duplicate_views_preserve_regularization(self):
        # If both views are identical, half weighting must recover the clean fit.
        rng = np.random.default_rng(3)
        x = rng.normal(size=(40, 12))
        y = np.tile(CATEGORIES, 4)
        clean, _, _ = fit_classifier(x, y, x, y, [1])
        aug, _, _ = fit_classifier(np.tile(x, (2, 1)), np.tile(y, 2), x, y, [1], np.full(80, .5))
        np.testing.assert_allclose(clean.coef_, aug.coef_, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
