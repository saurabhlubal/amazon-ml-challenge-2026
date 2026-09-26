"""
Post-processing pipeline for SageMaker Amazon ML Challenge 2026.
Waits for the SageMaker processing job to complete, downloads output shards,
merges them into final submission files, and runs the official submission validator.
"""

import os
import sys
import json
import time
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AWS_BIN = r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"
BUCKET = "sagemaker-amazon-ml-016933545204-ap-south-1"
PREFIX = "entity_resolution"
JOB_NAME = "amazon-ml-er-prod-2x-t3xlarge-1790448806"
S3_OUTPUT_URI = f"s3://{BUCKET}/{PREFIX}/output"
LOCAL_SHARDS_DIR = os.path.join(PROJECT_ROOT, "output", "sagemaker_shards")
LOCAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
TEST_DIR = os.path.join(PROJECT_ROOT, "student_resource", "dataset", "test")

def get_job_status():
    p = subprocess.run([AWS_BIN, "sagemaker", "describe-processing-job", "--processing-job-name", JOB_NAME], capture_output=True, text=True)
    if p.returncode != 0:
        return "Unknown", p.stderr
    data = json.loads(p.stdout)
    return data.get("ProcessingJobStatus", "Unknown"), data.get("FailureReason")

def sync_and_merge():
    print("\n" + "=" * 70)
    print(f"[Step 1/3] Downloading output shards from {S3_OUTPUT_URI}...")
    print("=" * 70)
    os.makedirs(LOCAL_SHARDS_DIR, exist_ok=True)
    subprocess.run([AWS_BIN, "s3", "sync", S3_OUTPUT_URI, LOCAL_SHARDS_DIR], check=True)
    
    # List downloaded shards
    shards = [f for f in os.listdir(LOCAL_SHARDS_DIR) if f.endswith(".tsv")]
    print(f"Downloaded {len(shards)} shard files:")
    for s in sorted(shards):
        fsize = os.path.getsize(os.path.join(LOCAL_SHARDS_DIR, s))
        print(f"  - {s} ({fsize / (1024**2):.2f} MB)")

    print("\n" + "=" * 70)
    print("[Step 2/3] Merging shards into deterministic final submission TSVs...")
    print("=" * 70)
    from sagemaker.merge_submission import merge_shards
    merge_shards(shard_dir=LOCAL_SHARDS_DIR, output_dir=LOCAL_OUTPUT_DIR, test_dir=TEST_DIR, validate=False)

    final_cand = os.path.join(LOCAL_OUTPUT_DIR, "candidate_pairs.tsv")
    final_match = os.path.join(LOCAL_OUTPUT_DIR, "matching_results.tsv")

    print("\n" + "=" * 70)
    print("[Step 3/3] Running official submission validator...")
    print("=" * 70)
    val_script = os.path.join(PROJECT_ROOT, "student_resource", "utils", "validate_submission.py")
    cmd = [
        sys.executable,
        val_script,
        "--matching", final_match,
        "--candidate", final_cand,
        "--test-dir", TEST_DIR,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    return res.returncode == 0

def main():
    status, fail_reason = get_job_status()
    print(f"Job Name : {JOB_NAME}")
    print(f"Status   : {status}")

    if status == "InProgress":
        print("\nThe SageMaker processing job is currently running.")
        print("To wait and finalize automatically when done, run this script.")
        return

    if status == "Completed":
        success = sync_and_merge()
        if success:
            print("\nPIPELINE EXECUTION COMPLETE & VALIDATED SUCCESSFULLY.")
        else:
            print("\nValidation failed!")
        return

    if status == "Failed":
        print(f"\nJob Failed! Reason: {fail_reason}")
        return

if __name__ == "__main__":
    main()
