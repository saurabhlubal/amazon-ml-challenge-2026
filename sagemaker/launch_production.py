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
INST_TYPE = "ml.c5.4xlarge"
INST_COUNT = 4
JOB_NAME = f"amazon-ml-er-c5-4xlarge-{int(time.time())}"

def main():
    print("=" * 70)
    print("RELAUNCHING SAGEMAKER PRODUCTION JOB")
    print(f"Cluster: {INST_COUNT}x {INST_TYPE} in {REGION}")
    print(f"Target S3: s3://{BUCKET}/{PREFIX}")
    print(f"Job Name: {JOB_NAME}")
    print("=" * 70)

    # 1. Package updated source code
    tar_path = os.path.join(PROJECT_ROOT, "sourcedir.tar.gz")
    print("\n[Step 1/3] Packaging updated source code with import fix...")
    with tarfile.open(tar_path, "w:gz") as tar:
        for folder in ["business_entity_resolution", "sagemaker", "scripts"]:
            src = os.path.join(PROJECT_ROOT, folder)
            tar.add(src, arcname=folder)
    print(f"  Created {tar_path} ({os.path.getsize(tar_path):,} bytes)")

    # 2. Upload updated code tarball to S3
    print("\n[Step 2/3] Uploading updated code tarball to S3...")
    s3_code_uri = f"s3://{BUCKET}/{PREFIX}/code/sourcedir.tar.gz"
    p_up = subprocess.run([AWS_BIN, "s3", "cp", tar_path, s3_code_uri], capture_output=True, text=True)
    if p_up.returncode != 0:
        print("Upload failed:", p_up.stderr)
        sys.exit(1)
    print("  Code tarball successfully uploaded to S3.")

    # 3. Create job specification for 4x ml.c5.4xlarge
    print(f"\n[Step 3/3] Submitting production processing job: {JOB_NAME}...")
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
        "AppSpecification": {
            "ImageUri": f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker",
            "ContainerEntrypoint": [
                "sh",
                "-c",
                f"cd /opt/ml/processing/input/code && tar -xzf sourcedir.tar.gz && python3 sagemaker/entrypoint.py --total-shards {INST_COUNT} --batch-size 2000"
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

    config_path = os.path.join(PROJECT_ROOT, "sagemaker", "c5_4xlarge_job.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(job_spec, f, indent=2)

    config_uri = "file://" + config_path.replace("\\", "/")
    p_sub = subprocess.run([AWS_BIN, "sagemaker", "create-processing-job", "--cli-input-json", config_uri], capture_output=True, text=True)

    if p_sub.returncode != 0:
        print("\nAWS Submission Error:")
        print(p_sub.stderr)
        return False, JOB_NAME, p_sub.stderr

    p_desc = subprocess.run([AWS_BIN, "sagemaker", "describe-processing-job", "--processing-job-name", JOB_NAME], capture_output=True, text=True)
    status_data = json.loads(p_desc.stdout)
    job_status = status_data.get("ProcessingJobStatus", "InProgress")

    print("\n" + "=" * 50)
    print("PRODUCTION JOB SUBMITTED")
    print(f"Job Name: {JOB_NAME}")
    print(f"Status: {job_status}")
    print(f"Cluster: {INST_COUNT}x {INST_TYPE}")
    print(f"Region: {REGION}")
    print(f"Output S3: s3://{BUCKET}/{PREFIX}/output")
    print("=" * 50)
    return True, JOB_NAME, job_status

if __name__ == "__main__":
    success, jname, detail = main()
    if not success:
        sys.exit(1)
