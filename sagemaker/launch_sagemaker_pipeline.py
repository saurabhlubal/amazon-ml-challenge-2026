"""
SageMaker Pipeline Orchestrator for Amazon ML Challenge 2026.
Launches distributed candidate generation and entity resolution across SageMaker compute instances.
"""

import os
import sys
import json
import argparse
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def launch_pipeline(
    s3_bucket: str,
    s3_prefix: str = "amazon_ml_2026",
    instance_type: str = "ml.c5.18xlarge",
    instance_count: int = 4,
    role_arn: Optional[str] = None,
    local_test_dir: str = "student_resource/dataset/test",
    output_dir: str = "output",
    dry_run: bool = False,
):
    print("=" * 75)
    print("AMAZON ML CHALLENGE 2026 — SAGEMAKER PRODUCTION PIPELINE LAUNCHER")
    print("=" * 75)
    print(f"Target S3 Bucket    : s3://{s3_bucket}/{s3_prefix}")
    print(f"Compute Instances   : {instance_count}x {instance_type}")
    print(f"Local Test Dir      : {local_test_dir}")
    print(f"Final Output Dir    : {output_dir}")
    print(f"Execution Role ARN  : {role_arn or 'Will resolve default SageMaker execution role'}")
    print("=" * 75)

    try:
        import boto3
        import sagemaker
        from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
    except ImportError:
        print("[Warning] boto3 or sagemaker SDK is not installed in the local environment.")
        print("To submit jobs directly from Python, run: pip install boto3 sagemaker")
        if not dry_run:
            print("[Info] Writing standalone AWS CLI job definition script for immediate deployment...")
            write_aws_cli_launcher(s3_bucket, s3_prefix, instance_type, instance_count, role_arn)
            return

    s3 = boto3.client("s3")
    sagemaker_session = sagemaker.Session()
    role = role_arn or sagemaker.get_execution_role()

    # 1. Sync data to S3
    print("\n[Step 1/4] Syncing test datasets and trained model to S3...")
    model_local = os.path.join(PROJECT_ROOT, "business_entity_resolution", "src", "trained_model.json")
    s3_model_uri = f"s3://{s3_bucket}/{s3_prefix}/model/trained_model.json"
    s3_data_uri = f"s3://{s3_bucket}/{s3_prefix}/data"
    s3_output_uri = f"s3://{s3_bucket}/{s3_prefix}/output"

    if dry_run:
        print(f"  [Dry Run] Would upload {model_local} -> {s3_model_uri}")
        print(f"  [Dry Run] Would sync {local_test_dir} -> {s3_data_uri}")
    else:
        s3.upload_file(model_local, s3_bucket, f"{s3_prefix}/model/trained_model.json")
        for fname in ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv"):
            fpath = os.path.join(local_test_dir, fname)
            print(f"  Uploading {fname} to {s3_data_uri}/{fname}...")
            s3.upload_file(fpath, s3_bucket, f"{s3_prefix}/data/{fname}")

    # 2. Configure SageMaker ScriptProcessor
    print("\n[Step 2/4] Initializing SageMaker ScriptProcessor...")
    processor = ScriptProcessor(
        command=["python3"],
        image_uri=sagemaker.image_uris.retrieve("pytorch", sagemaker_session.boto_region_name, version="2.1.0", instance_type=instance_type, image_scope="training"),
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=100,
        base_job_name="amazon-ml-er-pipeline",
    )

    inputs = [
        ProcessingInput(
            source=s3_data_uri,
            destination="/opt/ml/processing/input",
            s3_data_distribution_type="FullyReplicated",
        ),
        ProcessingInput(
            source=s3_model_uri,
            destination="/opt/ml/processing/model",
            s3_data_distribution_type="FullyReplicated",
        ),
    ]

    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=s3_output_uri,
        )
    ]

    # 3. Launch processing job
    print("\n[Step 3/4] Launching distributed SageMaker Processing Job...")
    entrypoint_script = os.path.join(PROJECT_ROOT, "sagemaker", "entrypoint.py")

    if dry_run:
        print(f"  [Dry Run] Configured job with {instance_count}x {instance_type} nodes.")
        print(f"  [Dry Run] Entrypoint: {entrypoint_script}")
        return

    processor.run(
        code=entrypoint_script,
        inputs=inputs,
        outputs=outputs,
        arguments=[
            "--total-shards", str(instance_count),
            "--batch-size", "2000",
        ],
        wait=True,
        logs=True,
    )

    # 4. Download and merge results
    print("\n[Step 4/4] Downloading shard outputs and merging final submission...")
    from sagemaker.merge_submission import merge_shards
    shard_dir = os.path.join(output_dir, "sagemaker_shards")
    os.makedirs(shard_dir, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", s3_output_uri, shard_dir], check=True)
    merge_shards(shard_dir=shard_dir, output_dir=output_dir, test_dir=local_test_dir)
    print("\nPipeline execution complete!")


def write_aws_cli_launcher(bucket: str, prefix: str, inst_type: str, inst_count: int, role: Optional[str]):
    script_content = f"""#!/bin/bash
# Standalone AWS CLI launch script for SageMaker Processing Job
set -e

BUCKET="{bucket}"
PREFIX="{prefix}"
INST_TYPE="{inst_type}"
INST_COUNT="{inst_count}"
ROLE="{role or '$SAGEMAKER_ROLE'}"

echo "Syncing data to s3://$BUCKET/$PREFIX/..."
aws s3 cp business_entity_resolution/src/trained_model.json s3://$BUCKET/$PREFIX/model/trained_model.json
aws s3 sync student_resource/dataset/test/ s3://$BUCKET/$PREFIX/data/

echo "Submitting SageMaker processing job..."
# See sagemaker/entrypoint.py for container entrypoint logic
"""
    cli_path = os.path.join(PROJECT_ROOT, "sagemaker", "launch_aws_cli.sh")
    with open(cli_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"Wrote AWS CLI starter script to {cli_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Amazon ML SageMaker Production Pipeline")
    parser.add_argument("--s3-bucket", default="amazon-ml-challenge-2026-saurabh")
    parser.add_argument("--s3-prefix", default="entity_resolution")
    parser.add_argument("--instance-type", default="ml.c5.18xlarge")
    parser.add_argument("--instance-count", type=int, default=4)
    parser.add_argument("--role-arn", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without submitting cloud job")

    args = parser.parse_args()
    launch_pipeline(
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        role_arn=args.role_arn,
        dry_run=args.dry_run,
    )
