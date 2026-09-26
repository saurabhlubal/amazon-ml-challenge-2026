"""
Unit tests for business_entity_resolution/src/features.py.
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import numpy as np
from business_entity_resolution.src.features import (
    FEATURE_NAMES,
    build_features,
)


class TestFeatures(unittest.TestCase):
    def test_feature_count_and_keys(self):
        """Verify that build_features returns exactly the expected 48 feature keys."""
        r1 = {
            "entity_id": "S1-100",
            "business_name": "Apex Technology Corp",
            "business_address": "1200 Market Street, Suite 400, Philadelphia, PA",
            "country": "US",
        }
        r2 = {
            "entity_id": "S2-200",
            "business_name": "Apex Technology Corporation",
            "business_address": "1200 Market St, Philadelphia, PA",
            "country": "US",
        }
        feats = build_features(r1, r2)
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        self.assertEqual(len(FEATURE_NAMES), 48)
        self.assertEqual(set(feats.keys()), set(FEATURE_NAMES))
        for k, v in feats.items():
            self.assertIsInstance(v, (int, float), f"Feature {k} is not numeric: {v}")
            self.assertFalse(np.isnan(v), f"Feature {k} is NaN")
            self.assertFalse(np.isinf(v), f"Feature {k} is Inf")

    def test_identical_records(self):
        """Verify that identical records yield 1.0 for all similarity features and 0.0 for diffs."""
        rec = {
            "entity_id": "S1-1",
            "business_name": "Acme Industrial Supplies LLC",
            "business_address": "450 Industrial Parkway, Dallas, TX",
            "country": "US",
        }
        f = build_features(rec, rec)
        self.assertEqual(f["name_exact_match"], 1.0)
        self.assertEqual(f["address_exact_match"], 1.0)
        self.assertEqual(f["country_exact_match"], 1.0)
        self.assertEqual(f["country_mismatch"], 0.0)
        self.assertEqual(f["name_levenshtein_sim"], 1.0)
        self.assertEqual(f["name_token_jaccard"], 1.0)
        self.assertEqual(f["name_char_3gram_jaccard"], 1.0)
        self.assertEqual(f["address_char_3gram_jaccard"], 1.0)
        self.assertEqual(f["name_length_diff"], 0.0)
        self.assertEqual(f["name_length_ratio"], 1.0)
        self.assertEqual(f["num_primary_match"], 1.0)

    def test_legal_and_street_normalization(self):
        """Verify that legal abbreviations and street abbreviations are standardized."""
        r1 = {
            "business_name": "Delta Logistics Corporation",
            "business_address": "100 South Boulevard, Seattle, WA",
            "country": "US",
        }
        r2 = {
            "business_name": "Delta Logistics Corp.",
            "business_address": "100 South Blvd, Seattle, WA",
            "country": "US",
        }
        f = build_features(r1, r2)
        # Normalized token Jaccard should be 1.0
        self.assertAlmostEqual(f["name_norm_token_jaccard"], 1.0, places=5)
        self.assertAlmostEqual(f["address_norm_token_jaccard"], 1.0, places=5)
        # Core tokens without legal suffix should also match perfectly
        self.assertAlmostEqual(f["name_core_token_jaccard"], 1.0, places=5)

        # French legal & street normalization test
        rf1 = {
            "business_name": "Thermal & Fils SASU",
            "business_address": "20 Boulevard Saint-Germain, Paris",
            "country": "France",
        }
        rf2 = {
            "business_name": "Thermal et Fils SAS",
            "business_address": "20 Blvd Saint-Germain, Paris",
            "country": "France",
        }
        ff = build_features(rf1, rf2)
        self.assertAlmostEqual(ff["name_norm_token_jaccard"], 1.0, places=5)
        self.assertAlmostEqual(ff["address_norm_token_jaccard"], 1.0, places=5)
        self.assertGreater(ff["name_char_3gram_jaccard"], 0.6)

    def test_missing_address_handling(self):
        """Verify that missing addresses set flags and default similarities safely to 0.0."""
        r1 = {
            "business_name": "Global Trade Center",
            "business_address": "",
            "country": "India",
        }
        r2 = {
            "business_name": "Global Trade Center Pvt Ltd",
            "business_address": "MG Road, Bengaluru",
            "country": "India",
        }
        f = build_features(r1, r2)
        self.assertEqual(f["address_is_missing_s1"], 1.0)
        self.assertEqual(f["address_is_missing_s2"], 0.0)
        self.assertEqual(f["address_is_missing_either"], 1.0)
        self.assertEqual(f["address_is_missing_both"], 0.0)
        self.assertEqual(f["address_levenshtein_sim"], 0.0)
        self.assertEqual(f["address_token_jaccard"], 0.0)

    def test_numeric_conflict_detection(self):
        """Verify that conflicting house/unit numbers produce high conflict flags."""
        r1 = {
            "business_name": "Cafe Mocha",
            "business_address": "104 Baker Street, London",
            "country": "UK",
        }
        r2 = {
            "business_name": "Cafe Mocha",
            "business_address": "208 Baker Street, London",
            "country": "UK",
        }
        f = build_features(r1, r2)
        self.assertEqual(f["num_address_conflict"], 1.0)
        self.assertEqual(f["num_primary_match"], 0.0)

    def test_country_mismatch(self):
        """Verify country mismatch detection."""
        r1 = {"business_name": "Starbucks", "business_address": "123 Main St", "country": "US"}
        r2 = {"business_name": "Starbucks", "business_address": "123 Main St", "country": "India"}
        f = build_features(r1, r2)
        self.assertEqual(f["country_exact_match"], 0.0)
        self.assertEqual(f["country_mismatch"], 1.0)
        self.assertEqual(f["name_and_country_match"], 0.0)

    def test_unicode_and_multilingual_handling(self):
        """Verify that Devanagari (Hindi) and accented characters are processed smoothly without crashing."""
        r1 = {
            "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "business_address": "KH NO. 570/13, NEW DELHI",
            "country": "India",
        }
        r2 = {
            "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
            "business_address": "KH NO. 570/13, NEW DELHI, Delhi",
            "country": "India",
        }
        f = build_features(r1, r2)
        self.assertEqual(f["name_exact_match"], 1.0)
        self.assertAlmostEqual(f["name_levenshtein_sim"], 1.0)
        self.assertGreater(f["address_token_jaccard"], 0.5)

        # French test set accented characters
        r3 = {
            "business_name": "Café de l'Étoile SARL",
            "business_address": "15 Avenue des Champs-Élysées, Paris",
            "country": "France",
        }
        r4 = {
            "business_name": "Cafe de l Etoile",
            "business_address": "15 Av des Champs Elysees, Paris",
            "country": "France",
        }
        f2 = build_features(r3, r4)
        self.assertGreater(f2["name_token_sort_sim"], 0.8)
        self.assertEqual(f2["country_exact_match"], 1.0)

    def test_none_and_empty_edge_cases(self):
        """Verify graceful handling when fields are None, empty strings, or numbers."""
        r_empty = {"business_name": None, "business_address": None, "country": None}
        r_num = {"business_name": 12345, "business_address": 67890, "country": "US"}
        f = build_features(r_empty, r_num)
        self.assertEqual(f["name_exact_match"], 0.0)
        self.assertEqual(f["country_is_missing"], 1.0)
        for k, v in f.items():
            self.assertFalse(np.isnan(v))


if __name__ == "__main__":
    unittest.main()
