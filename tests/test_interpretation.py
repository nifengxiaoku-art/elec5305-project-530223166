"""Checks for matched diagnostic counts and independent metric aggregation."""
import unittest
import numpy as np
import pandas as pd
from src.dataset import CATEGORIES
from src.deepen_analysis import matched_transitions, manual_macro_f1, leave_one_fold_out, transition_summary


class InterpretationTests(unittest.TestCase):
    def frames(self):
        a = pd.DataFrame({"filename":["a","b","c","d"],"true_label":["dog"]*4,
                          "predicted_label":["cat","dog","cat","dog"]})
        b = a.copy()
        b["predicted_label"] = ["dog","cat","cow","dog"]
        return a,b

    def test_transitions_and_counts(self):
        a,b = self.frames()
        result = matched_transitions(a,b,["filename"])
        self.assertEqual(result.transition.tolist(),["repaired","harmed","persistent_error","stable_correct"])
        repeated = pd.concat([result.assign(seed=s) for s in [17,29,43]])
        summary = transition_summary(repeated)
        self.assertEqual(summary["repaired_instances"],3)
        self.assertEqual(summary["repaired_unique_clips"],1)
        self.assertEqual(summary["net_correct_instances"],0)

    def test_alignment_fails_on_missing_duplicate_or_inconsistent_truth(self):
        a,b = self.frames()
        with self.assertRaises(ValueError):
            matched_transitions(a,b.iloc[:3],["filename"])
        with self.assertRaises(ValueError):
            matched_transitions(pd.concat([a,a.iloc[:1]]),b,["filename"])
        b.loc[0,"true_label"]="cat"
        with self.assertRaises(ValueError):
            matched_transitions(a,b,["filename"])

    def test_manual_metric_known_case(self):
        true = np.repeat(CATEGORIES,8)
        predicted = true.copy()
        predicted[np.where(true == "hen")[0][:6]]="rooster"
        predicted[np.where(true == "pig")[0][0]]="rooster"
        self.assertAlmostEqual(manual_macro_f1(true,predicted),.9028985507246376)

    def test_descriptive_fold_omission(self):
        values = pd.Series([0,0,0,0,.02],index=range(1,6))
        result = leave_one_fold_out(values)
        self.assertEqual(result[5],0)
        self.assertAlmostEqual(result[1],.005)
        with self.assertRaises(ValueError):
            leave_one_fold_out(values.iloc[:4])


if __name__ == "__main__":
    unittest.main()
