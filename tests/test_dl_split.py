import unittest

import numpy as np
from sklearn.model_selection import train_test_split

from analysis.run_dl_case_study import RANDOM_SEED, split_train_val


class HoldoutIdentityTests(unittest.TestCase):
    """The CNN must be scored on exactly the rows the Random Forest was scored on."""

    def setUp(self):
        self.labels = np.repeat(np.arange(6), 300)  # NEU-CLS: 6 balanced classes, 1800 images
        self.paths = [f"img_{i}" for i in range(len(self.labels))]

    def test_index_split_matches_run_neu_case_study_split(self):
        x = np.zeros((len(self.labels), 3))  # RF script splits features, labels and paths together
        _, _, _, _, _, rf_test_paths = train_test_split(
            x, self.labels, self.paths, test_size=0.20, random_state=RANDOM_SEED, stratify=self.labels
        )
        _, test_idx = train_test_split(
            np.arange(len(self.labels)), test_size=0.20, random_state=RANDOM_SEED, stratify=self.labels
        )
        self.assertEqual([self.paths[i] for i in test_idx], list(rf_test_paths))
        self.assertEqual(len(test_idx), 360)

    def test_validation_split_never_touches_the_test_rows(self):
        train_idx, test_idx = train_test_split(
            np.arange(len(self.labels)), test_size=0.20, random_state=RANDOM_SEED, stratify=self.labels
        )
        fit_idx, val_idx = split_train_val(train_idx, self.labels)
        self.assertEqual((len(fit_idx), len(val_idx)), (1296, 144))
        self.assertFalse(set(val_idx) & set(test_idx))
        self.assertFalse(set(fit_idx) & set(test_idx))
        self.assertFalse(set(fit_idx) & set(val_idx))
        self.assertEqual(np.bincount(self.labels[val_idx]).tolist(), [24] * 6)


if __name__ == "__main__":
    unittest.main()
