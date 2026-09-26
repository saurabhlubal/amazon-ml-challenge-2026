"""
SageMaker Pipeline Orchestrator for Amazon ML Challenge 2026.
Launches distributed candidate generation and entity resolution across SageMaker compute instances.
"""

import os
import sys
import json
import argparse
import subprocess
from typing import Optional, List, Dict

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
        from sagemaker.processing import ProcessingInput, ProcessingOutput
        from sagemaker.pytorch.processing import PyTorchProcessor
    except ImportError:
        print("[Warning] boto3 or sagemaker SDK is not installed in the local environment.")
        print("To submit jobs directly from Python, run: pip install boto3 sagemaker")
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

    # 2. Configure SageMaker PyTorchProcessor
    print("\n[Step 2/4] Initializing SageMaker PyTorchProcessor...")
    processor = PyTorchProcessor(
        framework_version="2.1.0",
        py_version="py310",
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        volume_size_in_gb=100,
        base_job_name="amazon-ml-er-pipeline",
        sagemaker_session=sagemaker_session,
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
    if dry_run:
        print(f"  [Dry Run] Configured job with {instance_count}x {instance_type} nodes.")
        print(f"  [Dry Run] Source dir packaged: {PROJECT_ROOT}")
        print(f"  [Dry Run] Entrypoint: sagemaker/entrypoint.py")
        return

    processor.run(
        code="sagemaker/entrypoint.py",
        source_dir=PROJECT_ROOT,
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
    resolved_role = role or "${SAGEMAKER_ROLE:-arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole}"
    
    # Bash version
    bash_script = f"""#!/bin/bash
# Standalone AWS CLI launch script for Amazon ML Challenge 2026 SageMaker Processing Job
set -euo pipefail

BUCKET="{bucket}"
PREFIX="{prefix}"
INST_TYPE="{inst_type}"
INST_COUNT={inst_count}
ROLE="{resolved_role}"
REGION="${{AWS_DEFAULT_REGION:-us-east-1}}"
JOB_NAME="amazon-ml-er-pipeline-$(date +%s)"

echo "=== [1/4] Syncing Data and Artifacts to S3 ==="
aws s3 cp business_entity_resolution/src/trained_model.json "s3://$BUCKET/$PREFIX/model/trained_model.json"
aws s3 cp student_resource/dataset/test/test_source1.tsv "s3://$BUCKET/$PREFIX/data/test_source1.tsv"
aws s3 cp student_resource/dataset/test/test_source2.tsv "s3://$BUCKET/$PREFIX/data/test_source2.tsv"
aws s3 cp student_resource/dataset/test/test_source3.tsv "s3://$BUCKET/$PREFIX/data/test_source3.tsv"

echo "=== [2/4] Packaging Source Code ==="
tar -czf /tmp/sourcedir.tar.gz business_entity_resolution sagemaker scripts
aws s3 cp /tmp/sourcedir.tar.gz "s3://$BUCKET/$PREFIX/code/sourcedir.tar.gz"

echo "=== [3/4] Launching SageMaker Processing Job: $JOB_NAME ==="
aws sagemaker create-processing-job \\
    --processing-job-name "$JOB_NAME" \\
    --role-arn "$ROLE" \\
    --processing-resources "ClusterConfig={{InstanceCount=$INST_COUNT,InstanceType=$INST_TYPE,VolumeSizeInGB=100}}" \\
    --app-specification "ImageUri=763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker,ContainerEntrypoint=['python3','sagemaker/entrypoint.py','--total-shards','$INST_COUNT','--batch-size','2000']" \\
    --processing-inputs "[{{\\"InputName\\":\\"data\\",\\"S3Input\\":{{\\"S3Uri\\":\\"s3://$BUCKET/$PREFIX/data\\",\\"LocalPath\\":\\"/opt/ml/processing/input\\",\\"S3DataType\\":\\"S3Prefix\\",\\"S3InputMode\\":\\"File\\",\\"S3DataDistributionType\\":\\"FullyReplicated\\"}}}},{{\\"InputName\\":\\"model\\",\\"S3Input\\":{{\\"S3Uri\\":\\"s3://$BUCKET/$PREFIX/model\\",\\"LocalPath\\":\\"/opt/ml/processing/model\\",\\"S3DataType\\":\\"S3Prefix\\",\\"S3InputMode\\":\\"File\\",\\"S3DataDistributionType\\":\\"FullyReplicated\\"}}}},{{\\"InputName\\":\\"code\\",\\"S3Input\\":{{\\"S3Uri\\":\\"s3://$BUCKET/$PREFIX/code/sourcedir.tar.gz\\",\\"LocalPath\\":\\"/opt/ml/processing/input/code\\",\\"S3DataType\\":\\"S3Prefix\\",\\"S3InputMode\\":\\"File\\",\\"S3CompressionType\\":\\"Gzip\\"}}开发}}]" \\
    --processing-output-config "Outputs=[{{\\"OutputName\\":\\"output\\",\\"S3Output\\":{{\\"S3Uri\\":\\"s3://$BUCKET/$PREFIX/output\\",\\"LocalPath\\":\\"/opt/ml/processing/output\\",\\"S3UploadMode\\":\\"EndOfJob\\"}}开发}}]"

echo "=== Job submitted successfully. Waiting for completion... ==="
aws sagemaker wait processing-job-completed-or-stopped --processing-job-name "$JOB_NAME"

echo "=== [4/4] Syncing Shards and Merging Final Submission ==="
mkdir -p output/sagemaker_shards
aws s3 sync "s3://$BUCKET/$PREFIX/output" output/sagemaker_shards
python sagemaker/merge_submission.py --shard-dir output/sagemaker_shards --output-dir output --test-dir student_resource/dataset/test
"""
    # Fix escaping artifact in bash script
    bash_script = bash_script.replace("开发", "")
    cli_path = os.path.join(PROJECT_ROOT, "sagemaker", "launch_aws_cli.sh")
    with open(cli_path, "w", encoding="utf-8") as f:
        f.write(bash_script)
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
