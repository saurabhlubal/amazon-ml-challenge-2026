"""
End-to-End Pipeline Orchestrator for Amazon ML Challenge 2026.
Integrates candidate generation (blocking), feature extraction, model scoring,
decision logic, and output validation.

Supports:
- Streaming / chunked data processing for large-scale datasets (~10M records)
- Strict compliance with student_resource/utils/validate_submission.py
- Pluggable teammate modules (normalization, blocking, features, model)
- Mock / smoke-test mode for verifying the entire pipeline scaffolding
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Set, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline_utils import (
    DELIM,
    MATCHING_HEADER,
    CANDIDATE_HEADER,
    format_id_list,
    stream_tsv_records,
    validate_submission_files,
)

# Shared interfaces
from business_entity_resolution.src.normalization import normalize_record
from business_entity_resolution.src.blocking import generate_candidates
from business_entity_resolution.src.features import build_features
from business_entity_resolution.src.model import predict_scores, decide_matches


class MockComponents:
    """Fallback mock implementations for pipeline testing before teammate modules are ready."""

    @staticmethod
    def normalize_record(record: Dict[str, str]) -> Dict[str, Any]:
        return {
            "entity_id": record.get("entity_id", ""),
            "business_name": record.get("business_name", "").strip().lower(),
            "business_address": record.get("business_address", "").strip().lower(),
            "country": record.get("country", "").strip().upper(),
        }

    @staticmethod
    def generate_candidates(source1_record: Dict[str, str], indexes: Dict[str, Any]) -> Set[str]:
        # Return candidate IDs that match exact normalized name or country
        norm_name = source1_record.get("business_name", "").strip().lower()
        candidates = set(indexes.get("name_index", {}).get(norm_name, []))
        return candidates

    @staticmethod
    def build_features(source1_record: Dict[str, str], candidate_record: Dict[str, str]) -> Dict[str, float]:
        s1_name = source1_record.get("business_name", "").strip().lower()
        cand_name = candidate_record.get("business_name", "").strip().lower()
        s1_country = source1_record.get("country", "").strip().upper()
        cand_country = candidate_record.get("country", "").strip().upper()
        name_match = 1.0 if s1_name and s1_name == cand_name else 0.0
        country_match = 1.0 if s1_country and s1_country == cand_country else 0.0
        return {"name_match": name_match, "country_match": country_match}

    @staticmethod
    def predict_scores(model: Any, X: List[Dict[str, float]]) -> List[float]:
        # Simple heuristic score based on feature average
        scores = []
        for feat in X:
            score = 0.7 * feat.get("name_match", 0.0) + 0.3 * feat.get("country_match", 0.0)
            scores.append(score)
        return scores

    @staticmethod
    def decide_matches(candidate_ids: List[str], scores: List[float], threshold: float) -> List[str]:
        return [cid for cid, s in zip(candidate_ids, scores) if s >= threshold]


def build_mock_indexes(s2_path: str, s3_path: str, max_records: Optional[int] = None) -> Dict[str, Any]:
    """Build mock indexes from Source2 and Source3 files."""
    name_index: Dict[str, List[str]] = {}
    record_store: Dict[str, Dict[str, str]] = {}
    count = 0

    for source_path in (s2_path, s3_path):
        if not os.path.isfile(source_path):
            continue
        for rec in stream_tsv_records(source_path, max_records=max_records):
            eid = rec.get("entity_id", "")
            norm_name = rec.get("business_name", "").strip().lower()
            if eid:
                record_store[eid] = rec
                if norm_name:
                    name_index.setdefault(norm_name, []).append(eid)
            count += 1
            if max_records and count >= max_records:
                break
    return {"name_index": name_index, "records": record_store}


def run_pipeline(
    test_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    max_records: Optional[int] = None,
    use_mock: bool = False,
    validate: bool = True,
    check_ids: bool = False,
) -> int:
    """
    Run the end-to-end entity resolution pipeline.
    """
    t0 = time.time()
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")

    if not os.path.isfile(s1_path):
        print(f"ERROR: {s1_path} not found.", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)
    candidate_out_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_out_path = os.path.join(output_dir, "matching_results.tsv")

    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026 - ENTITY RESOLUTION PIPELINE")
    print("=" * 70)
    print(f"Test directory     : {test_dir}")
    print(f"Output directory   : {output_dir}")
    print(f"Decision threshold : {threshold}")
    print(f"Mode               : {'MOCK / SMOKE-TEST' if use_mock else 'PRODUCTION (Shared Modules)'}")
    print(f"Max S1 records     : {max_records or 'ALL'}")
    print("-" * 70)

    # 1. Component Resolution
    if use_mock:
        fn_normalize = MockComponents.normalize_record
        fn_blocking = MockComponents.generate_candidates
        fn_features = MockComponents.build_features
        fn_predict = MockComponents.predict_scores
        fn_decide = MockComponents.decide_matches
        indexes = build_mock_indexes(s2_path, s3_path, max_records=max_records)
        model = None
    else:
        fn_normalize = normalize_record
        fn_blocking = generate_candidates
        fn_features = build_features
        fn_predict = predict_scores
        fn_decide = decide_matches
        indexes = {}
        model = None

    # 2. Process Source 1 and stream outputs
    total_s1 = 0
    total_candidates = 0
    total_matches = 0
    singletons = 0

    print("Streaming and processing records...")
    with open(candidate_out_path, "w", encoding="utf-8", newline="") as f_cand, \
         open(matching_out_path, "w", encoding="utf-8", newline="") as f_match:

        # Write required headers
        f_cand.write(CANDIDATE_HEADER)
        f_match.write(MATCHING_HEADER)

        for s1_rec in stream_tsv_records(s1_path, max_records=max_records):
            s1_id = s1_rec.get("entity_id", "")
            if not s1_id:
                continue
            total_s1 += 1

            # Optional normalization if implemented
            try:
                s1_proc = fn_normalize(s1_rec)
            except NotImplementedError:
                s1_proc = s1_rec

            # Candidate Generation
            try:
                candidates = fn_blocking(s1_proc, indexes)
            except NotImplementedError:
                print("ERROR: Shared blocking module is not implemented yet.", file=sys.stderr)
                print("Hint: Run with --mock to test pipeline scaffolding.", file=sys.stderr)
                return 1

            cand_list = sorted(list(candidates))
            total_candidates += len(cand_list)
            cand_str = format_id_list(cand_list)
            f_cand.write(f"{s1_id}{DELIM}{cand_str}\n")

            # Matching and Decision
            if not cand_list:
                f_match.write(f"{s1_id}{DELIM}\n")
                singletons += 1
                continue

            try:
                # Build pair features using record store if available
                records_store = indexes.get("records", {})
                X_pairs = [
                    fn_features(s1_proc, records_store.get(cid, {"entity_id": cid}))
                    for cid in cand_list
                ]

                scores = fn_predict(model, X_pairs)
                matched_ids = fn_decide(cand_list, scores, threshold)
            except NotImplementedError:
                print("ERROR: Shared feature/model modules are not implemented yet.", file=sys.stderr)
                print("Hint: Run with --mock to test pipeline scaffolding.", file=sys.stderr)
                return 1

            # Ensure matched IDs are a subset of candidates
            valid_matches = [m for m in matched_ids if m in candidates]
            match_str = format_id_list(valid_matches)
            f_match.write(f"{s1_id}{DELIM}{match_str}\n")

            if valid_matches:
                total_matches += len(valid_matches)
            else:
                singletons += 1

            if total_s1 % 100_000 == 0:
                print(f"Processed {total_s1:,} S1 records... ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    avg_cand = total_candidates / total_s1 if total_s1 else 0.0
    avg_match = total_matches / total_s1 if total_s1 else 0.0

    print("-" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("-" * 70)
    print(f"Total Source1 processed : {total_s1:,}")
    print(f"Total candidate pairs   : {total_candidates:,} (avg {avg_cand:.2f}/S1)")
    print(f"Total predicted matches : {total_matches:,} (avg {avg_match:.2f}/S1)")
    print(f"Singletons (0 matches)  : {singletons:,} ({singletons / total_s1 * 100:.2f}%)")
    print(f"Elapsed time            : {elapsed:.2f} seconds")
    print(f"Candidate file saved to : {candidate_out_path}")
    print(f"Matching file saved to  : {matching_out_path}")
    print("=" * 70)

    # 3. Post-run Validation
    if validate:
        print("\nRunning official submission validator...")
        val_code = validate_submission_files(
            matching_path=matching_out_path,
            candidate_path=candidate_out_path,
            test_dir=test_dir,
            check_ids=check_ids,
        )
        return val_code

    return 0


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline Orchestrator")
    parser.add_argument("--test-dir", default=os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test"),
                        help="Path to test directory containing test_source1/2/3.tsv")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output"),
                        help="Path to output directory for results")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for matching")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum S1 records to process (for testing/sampling)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock components for testing pipeline plumbing")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip post-run submission validation")
    parser.add_argument("--check-ids", action="store_true",
                        help="Run full ID-existence check in validator (memory-heavy)")
    args = parser.parse_args()

    exit_code = run_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_records=args.max_records,
        use_mock=args.mock,
        validate=not args.no_validate,
        check_ids=args.check_ids,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
