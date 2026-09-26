"""
Unit tests for business_entity_resolution.src.blocking.
"""

import unittest
from business_entity_resolution.src.blocking import (
    build_blocking_indexes,
    generate_candidates,
    batch_generate_candidates,
    evaluate_blocking,
)


class TestBlocking(unittest.TestCase):

    def setUp(self):
        self.candidates = [
            {
                "entity_id": "S2-001",
                "business_name": "Maure Williams Colombier",
                "business_address": "85 Wayne Ave, Ticonderoga, NY",
                "country": "US"
            },
            {
                "entity_id": "S3-001",
                "business_name": "maurewilliamscolombier.com",
                "business_address": "Wayne Ave, Ticonderoga, NY",
                "country": "US"
            },
            {
                "entity_id": "S2-002",
                "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
                "business_address": "KH NO. -570/13, NEW DELHI, Delhi",
                "country": "INDIA"
            },
            {
                "entity_id": "S3-002",
                "business_name": "Unrelated Business Corp",
                "business_address": "123 Main St, Austin, TX",
                "country": "US"
            },
        ]
        self.indexes = build_blocking_indexes(self.candidates)

    def test_build_indexes(self):
        self.assertIn("exact_core", self.indexes)
        self.assertIn("sorted_core", self.indexes)
        self.assertIn("compact", self.indexes)
        self.assertEqual(self.indexes["total_records"], 4)

    def test_generate_candidates_name_match(self):
        s1 = {
            "entity_id": "S1-100",
            "business_name": "Maure Williams Colombier Inc",
            "business_address": "",
            "country": "US"
        }
        cands = generate_candidates(s1, self.indexes)
        self.assertIn("S2-001", cands)
        self.assertIn("S3-001", cands)
        self.assertNotIn("S3-002", cands)
        self.assertNotIn("S2-002", cands)  # India should not match US

    def test_generate_candidates_transliterated(self):
        s1 = {
            "entity_id": "S1-200",
            "business_name": "Ram Marketing Private Limited",
            "business_address": "570/13 New Delhi, Delhi",
            "country": "INDIA"
        }
        cands = generate_candidates(s1, self.indexes)
        self.assertIn("S2-002", cands)
        self.assertNotIn("S2-001", cands)

    def test_batch_generate_candidates(self):
        s1_list = [
            {
                "entity_id": "S1-100",
                "business_name": "Maure Williams Colombier Inc",
                "business_address": "",
                "country": "US"
            }
        ]
        batch = batch_generate_candidates(s1_list, self.indexes)
        self.assertIn("S1-100", batch)
        self.assertIn("S2-001", batch["S1-100"])

    def test_evaluate_blocking(self):
        s1_records = {
            "S1-100": {
                "entity_id": "S1-100",
                "business_name": "Maure Williams Colombier Inc",
                "business_address": "85 Wayne Ave, Ticonderoga, NY",
                "country": "US"
            }
        }
        ground_truth = {
            "S1-100": {"S2-001", "S3-001"}
        }
        metrics = evaluate_blocking(s1_records, ground_truth, self.indexes)
        self.assertEqual(metrics["blocking_recall"], 1.0)
        self.assertEqual(metrics["found_true_matches"], 2)
        self.assertLessEqual(metrics["average_candidates"], 4.0)


if __name__ == "__main__":
    unittest.main()
