import unittest
from datetime import date

import pandas as pd

from pystocks.price_preprocess import PricePreprocessConfig, _compute_eligibility


class PricePreprocessEligibilityTests(unittest.TestCase):
    def test_missing_ratio_rule_is_enforced(self):
        df = pd.DataFrame(
            {
                "conid": ["A", "A"],
                "trade_date": [date(2024, 1, 1), date(2024, 1, 10)],
                "is_clean_price": [True, True],
            }
        )
        cfg = PricePreprocessConfig(
            min_history_days=2,
            max_missing_ratio=0.1,
            max_internal_gap_days=20,
        )
        out = _compute_eligibility(df, cfg)
        row = out.iloc[0]
        self.assertFalse(bool(row["eligible"]))
        self.assertIn("Excessive missing ratio", str(row["eligibility_reason"]))

    def test_internal_gap_rule_is_enforced(self):
        df = pd.DataFrame(
            {
                "conid": ["B", "B"],
                "trade_date": [date(2024, 1, 1), date(2024, 1, 5)],
                "is_clean_price": [True, True],
            }
        )
        cfg = PricePreprocessConfig(
            min_history_days=2,
            max_missing_ratio=1.0,
            max_internal_gap_days=1,
        )
        out = _compute_eligibility(df, cfg)
        row = out.iloc[0]
        self.assertFalse(bool(row["eligible"]))
        self.assertIn("Large internal gap", str(row["eligibility_reason"]))

    def test_eligible_when_all_rules_pass(self):
        df = pd.DataFrame(
            {
                "conid": ["C", "C", "C"],
                "trade_date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
                "is_clean_price": [True, True, True],
            }
        )
        cfg = PricePreprocessConfig(
            min_history_days=3,
            max_missing_ratio=0.5,
            max_internal_gap_days=5,
        )
        out = _compute_eligibility(df, cfg)
        row = out.iloc[0]
        self.assertTrue(bool(row["eligible"]))
        self.assertEqual(str(row["eligibility_reason"]), "OK")


if __name__ == "__main__":
    unittest.main()
