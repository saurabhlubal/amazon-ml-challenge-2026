import os
import sys
import json
import time
import tarfile
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AWS_BIN = r"C:\Program Files\Amazon\AWSCLIV2\aws.exe"
BUCKET = "sagemaker-amazon-ml-016933545204-ap-south-1"
PREFIX = "entity_resolution"
ROLE = "arn:aws:iam::016933545204:role/AmazonSageMaker-ExecutionRole-AmazonML"
REGION = "ap-south-1"
INST_TYPE = "ml.t3.xlarge"
INST_COUNT = 2
JOB_NAME = f"amazon-ml-er-prod-2x-t3xlarge-{int(time.time())}"

def main():
    print("=" * 70)
    print("SAGEMAKER PRODUCTION PIPELINE LAUNCHER (2x ml.t3.xlarge)")
    print("=" * 70)

    # 1. Package source code
    tar_path = os.path.join(PROJECT_ROOT, "sourcedir.tar.gz")
    print("\n[Step 1/3] Packaging updated source code...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for folder in ["business_entity_resolution", "sagemaker", "scripts"]:
            src = os.path.join(PROJECT_ROOT, folder)
            tar.add(src, arcname=folder)
    print(f"  Created {tar_path} ({os.path.getsize(tar_path):,} bytes)")

    # 2. Upload code tarball to S3
    print("\n[Step 2/3] Uploading updated code tarball to S3...")
    s3_code_uri = f"s3://{BUCKET}/{PREFIX}/code/sourcedir.tar.gz"
    p_up = subprocess.run([AWS_BIN, "s3", "cp", tar_path, s3_code_uri], capture_output=True, text=True)
    if p_up.returncode != 0:
        print("Upload failed:", p_up.stderr)
        sys.exit(1)
    print(f"  Uploaded to {s3_code_uri}")

    # 3. Build job specification
    job_spec = {
        "ProcessingJobName": JOB_NAME,
        "RoleArn": ROLE,
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": INST_COUNT,
                "InstanceType": INST_TYPE,
                "VolumeSizeInGB": 100
            }
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 86400
        },
        "AppSpecification": {
            "ImageUri": f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker",
            "ContainerEntrypoint": [
                "sh",
                "-c",
                f"cd /opt/ml/processing/input/code && tar -xzf sourcedir.tar.gz && python3 sagemaker/entrypoint.py --total-shards {INST_COUNT} --batch-size 2000 --cand-chunk-size 1250000"
            ]
        },
        "ProcessingInputs": [
            {
                "InputName": "data",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{PREFIX}/data",
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            {
                "InputName": "model",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{PREFIX}/model",
                    "LocalPath": "/opt/ml/processing/model",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated"
                }
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": f"s3://{BUCKET}/{PREFIX}/code/sourcedir.tar.gz",
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File"
                }
            }
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "output",
                    "S3Output": {
                        "S3Uri": f"s3://{BUCKET}/{PREFIX}/output",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob"
                    }
                }
            ]
        }
    }

    # Verify configuration fields
    cluster_cfg = job_spec["ProcessingResources"]["ClusterConfig"]
    print("\n[Step 3/3] Verifying Job Configuration Before Submission:")
    print(f"  InstanceType       : {cluster_cfg['InstanceType']}")
    print(f"  InstanceCount      : {cluster_cfg['InstanceCount']}")
    print(f"  VolumeSizeInGB     : {cluster_cfg['VolumeSizeInGB']}")
    print(f"  Region             : {REGION}")
    print(f"  Execution Role     : {ROLE}")
    print(f"  Job Name           : {JOB_NAME}")
    print(f"  Input Data URI     : s3://{BUCKET}/{PREFIX}/data")
    print(f"  Input Model URI    : s3://{BUCKET}/{PREFIX}/model")
    print(f"  Output S3 URI      : s3://{BUCKET}/{PREFIX}/output")
    print(f"  Container Command  : {' '.join(job_spec['AppSpecification']['ContainerEntrypoint'])}")

    assert cluster_cfg["InstanceType"] == "ml.t3.xlarge", f"Expected ml.t3.xlarge, got {cluster_cfg['InstanceType']}"
    assert cluster_cfg["InstanceCount"] == 2, f"Expected 2, got {cluster_cfg['InstanceCount']}"
    print("  >> Configuration Verified: InstanceType=ml.t3.xlarge and InstanceCount=2")

    config_path = os.path.join(PROJECT_ROOT, "sagemaker", "t3_2x_job_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(job_spec, f, indent=2)

    config_uri = "file://" + config_path.replace("\\", "/")
    print(f"\nSubmitting SageMaker Processing Job: {JOB_NAME}...")
    p_sub = subprocess.run([AWS_BIN, "sagemaker", "create-processing-job", "--cli-input-json", config_uri], capture_output=True, text=True)

    if p_sub.returncode != 0:
        print("\nAWS Submission Error:")
        print(p_sub.stderr)
        sys.exit(1)

    print(p_sub.stdout)

    p_desc = subprocess.run([AWS_BIN, "sagemaker", "describe-processing-job", "--processing-job-name", JOB_NAME], capture_output=True, text=True)
    status_data = json.loads(p_desc.stdout)
    job_status = status_data.get("ProcessingJobStatus", "InProgress")
    job_arn = status_data.get("ProcessingJobArn", "")

    print("\n" + "=" * 50)
    print("PRODUCTION JOB SUBMITTED")
    print(f"Job Name: {JOB_NAME}")
    print(f"Status: {job_status}")
    print(f"ARN: {job_arn}")
    print(f"Cluster: {cluster_cfg['InstanceCount']}x {cluster_cfg['InstanceType']}")
    print(f"Output S3: s3://{BUCKET}/{PREFIX}/output")
    print("=" * 50)

    # Clean up local tar and config
    if os.path.exists(tar_path):
        os.remove(tar_path)
    if os.path.exists(config_path):
        os.remove(config_path)

if __name__ == "__main__":
    main()
