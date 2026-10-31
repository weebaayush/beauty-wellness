"""Unit tests for the cleaning steps.  Run:  python -m unittest discover -s tests -v"""
import unittest

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.cleaning import steps
from src.cleaning.impute_outliers import robust_z
from src.cleaning.log import CleaningLog

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "cleaning.yaml").read_text())


class TestParsers(unittest.TestCase):
    def test_price(self):
        self.assertEqual(steps.parse_price("$42.00"), 42.0)
        self.assertEqual(steps.parse_price("USD 1,250"), 1250.0)
        self.assertTrue(np.isnan(steps.parse_price("free")))

    def test_size(self):
        self.assertEqual(steps.parse_size_ml("1.7 oz/ 50 mL"), 50.0)      # mL preferred when both present
        self.assertAlmostEqual(steps.parse_size_ml("1 oz"), 29.6, places=1)
        self.assertEqual(steps.parse_size_ml("75 ML"), 75.0)
        self.assertTrue(np.isnan(steps.parse_size_ml("travel size")))

    def test_brand_key(self):
        self.assertEqual(steps.canonical_brand(" PETAL AND PINE. "), steps.canonical_brand("Petal & Pine"))

    def test_lookup(self):
        lut = steps.build_lookup(CFG["mappings"]["skin_type"])
        self.assertEqual(lut[steps._norm_key(" Combo ")], "combination")


class TestText(unittest.TestCase):
    def test_clean_text(self):
        s = steps.clean_text("<p>Itâ€™s great &amp; cheap</p> mail me a.b@x.com https://x.io 😍")
        self.assertNotIn("<p>", s)
        self.assertIn("It's great & cheap", s)
        self.assertIn("[EMAIL]", s)
        self.assertIn("[URL]", s)
        self.assertIn("😍", s)          # emojis are kept on purpose

    def test_ingredients(self):
        syn = steps.build_lookup(CFG["mappings"]["ingredient_synonyms"])
        raw = ("-Niacinamide: Minimizes pores. AQUA, Glycerin, Butyrospermum Parkii (Shea) Butter, "
               "Parfum. Please be aware that ingredient lists may change.")
        self.assertEqual(steps.parse_ingredients(raw, syn), ["water", "glycerin", "shea butter", "fragrance"])

    def test_inci_split_respects_parentheses(self):
        self.assertEqual(len(steps.split_inci("Aqua (Water, Eau), Glycerin")), 2)


class TestRules(unittest.TestCase):
    def test_validity_and_duplicates(self):
        log = CleaningLog()
        prod = pd.DataFrame({"product_id": ["P1", "P1"], "price_usd": [9999.0, 20.0], "loves_count": [-5, 5]})
        rev = pd.DataFrame({"author_id": ["a", "a", "b"], "product_id": ["P1", "P1", "P1"], "rating": [6, 4, 1],
                            "is_recommended": [1, 1, 1], "helpfulness": [50.0, 0.5, np.nan],
                            "total_feedback_count": [2, 2, 0], "total_pos_feedback_count": [1, 1, 0],
                            "total_neg_feedback_count": [1, 1, 0], "review_text": ["Good", "Good", "Bad"],
                            "submission_time": pd.to_datetime(["2020-01-01", "2020-01-01", "2035-01-01"])})
        p, r = steps.apply_validity_rules(prod, rev, CFG, log)
        self.assertTrue(np.isnan(p.loc[0, "price_usd"]))
        self.assertEqual(p.loc[0, "loves_count"], 5)
        self.assertTrue(np.isnan(r.loc[0, "rating"]))
        self.assertEqual(r.loc[0, "helpfulness"], 0.5)
        self.assertTrue(pd.isna(r.loc[2, "submission_time"]))
        self.assertTrue(r.loc[2, "flag_rating_rec_conflict"])
        p2, r2 = steps.remove_duplicates(p, r, log)
        self.assertEqual(len(p2), 1)
        self.assertEqual(len(r2), 2)

    def test_robust_z_flags_extreme(self):
        z = robust_z(pd.Series([10, 11, 12, 10, 11, 200.0]))
        self.assertGreater(z.iloc[-1], 3.5)
        self.assertLess(abs(z.iloc[0]), 3.5)


if __name__ == "__main__":
    unittest.main()
