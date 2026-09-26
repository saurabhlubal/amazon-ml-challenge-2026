"""
Unit tests for business_entity_resolution.src.normalization.
"""

import unittest
from business_entity_resolution.src.normalization import (
    clean_text,
    repair_mojibake,
    strip_latin_accents,
    transliterate_indic,
    extract_domain_root,
    clean_digit,
    normalize_business_name,
    normalize_address,
    normalize_record,
)


class TestNormalization(unittest.TestCase):

    def test_mojibake_repair(self):
        text = "CafÃ© &amp; Restaurant â€œDeliciousâ€"
        cleaned = clean_text(text)
        self.assertIn("cafe", cleaned)
        self.assertIn("restaurant", cleaned)
        self.assertIn("delicious", cleaned)

    def test_latin_accents(self):
        text = "Société d'Électricité & Mécanique"
        cleaned = clean_text(text)
        self.assertEqual(cleaned, "societe d electricite mecanique")

    def test_indic_transliteration_hindi(self):
        text = "राम मार्केटिंग प्राइवेट लिमिटेड"
        cleaned = clean_text(text)
        self.assertIn("ram", cleaned)
        self.assertIn("marketing", cleaned)
        self.assertIn("pvt", cleaned)
        self.assertIn("ltd", cleaned)

    def test_indic_transliteration_gujarati(self):
        text = "વન ઇન્ફ્રા પ્રાઇવેટ લિમિટેડ"
        cleaned = clean_text(text)
        self.assertIn("infra", cleaned)
        self.assertIn("pvt", cleaned)
        self.assertIn("ltd", cleaned)

    def test_indic_transliteration_oriya(self):
        text = "ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍"
        cleaned = clean_text(text)
        self.assertIn("shakti", cleaned)
        self.assertIn("aagro", cleaned)

    def test_domain_extraction(self):
        self.assertEqual(extract_domain_root("bryansquare.com"), "bryansquare")
        self.assertEqual(extract_domain_root("kbmresearch.c0m"), "kbmresearch")
        self.assertEqual(extract_domain_root("https://www.technologiesmarketing.org"), "technologiesmarketing")
        self.assertIsNone(extract_domain_root("Regular Business Name"))

    def test_clean_digit(self):
        self.assertEqual(clean_digit("007"), "7")
        self.assertEqual(clean_digit("0356"), "356")
        self.assertEqual(clean_digit("1401"), "1401")
        self.assertEqual(clean_digit("000"), "0")

    def test_normalize_business_name(self):
        res = normalize_business_name("Maure Williams Colombier Inc.")
        self.assertEqual(res["clean_name"], "maure williams colombier inc")
        self.assertEqual(res["core_name"], "maure williams colombier")
        self.assertIn("maure", res["core_tokens"])
        self.assertIn("williams", res["core_tokens"])
        self.assertIn("colombier", res["core_tokens"])
        self.assertNotIn("inc", res["core_tokens"])
        self.assertIn("maurewilliamscolombier", res["compact_forms"])

    def test_normalize_address(self):
        res = normalize_address("141-08 71 Road, Flushing, NY")
        self.assertIn("141", res["digits"])
        self.assertIn("8", res["digits"])
        self.assertIn("71", res["digits"])
        self.assertIn("flushing", res["significant_tokens"])

    def test_normalize_record(self):
        rec = {
            "entity_id": "S1-965667",
            "business_name": "Maure Williams Colombier Inc",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY",
            "country": "US"
        }
        norm = normalize_record(rec)
        self.assertEqual(norm["entity_id"], "S1-965667")
        self.assertEqual(norm["country"], "US")
        self.assertEqual(norm["core_name"], "maure williams colombier")
        self.assertEqual(norm["first_digit"], "85")
        self.assertIn("ticonderoga", norm["significant_address_tokens"])


if __name__ == "__main__":
    unittest.main()
