#!/bin/bash
# Standalone AWS CLI launch script for Amazon ML Challenge 2026 SageMaker Processing Job
set -euo pipefail

BUCKET="amazon-ml-challenge-2026-saurabh"
PREFIX="entity_resolution"
INST_TYPE="ml.c5.4xlarge"
INST_COUNT=4
ROLE="${SAGEMAKER_ROLE:-arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole}"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
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
aws sagemaker create-processing-job \
    --processing-job-name "$JOB_NAME" \
    --role-arn "$ROLE" \
    --processing-resources "ClusterConfig={InstanceCount=$INST_COUNT,InstanceType=$INST_TYPE,VolumeSizeInGB=100}" \
    --app-specification "ImageUri=763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker,ContainerEntrypoint=['python3','sagemaker/entrypoint.py','--total-shards','$INST_COUNT','--batch-size','2000']" \
    --processing-inputs "[{\"InputName\":\"data\",\"S3Input\":{\"S3Uri\":\"s3://$BUCKET/$PREFIX/data\",\"LocalPath\":\"/opt/ml/processing/input\",\"S3DataType\":\"S3Prefix\",\"S3InputMode\":\"File\",\"S3DataDistributionType\":\"FullyReplicated\"}},{\"InputName\":\"model\",\"S3Input\":{\"S3Uri\":\"s3://$BUCKET/$PREFIX/model\",\"LocalPath\":\"/opt/ml/processing/model\",\"S3DataType\":\"S3Prefix\",\"S3InputMode\":\"File\",\"S3DataDistributionType\":\"FullyReplicated\"}},{\"InputName\":\"code\",\"S3Input\":{\"S3Uri\":\"s3://$BUCKET/$PREFIX/code/sourcedir.tar.gz\",\"LocalPath\":\"/opt/ml/processing/input/code\",\"S3DataType\":\"S3Prefix\",\"S3InputMode\":\"File\",\"S3CompressionType\":\"Gzip\"}}]" \
    --processing-output-config "Outputs=[{\"OutputName\":\"output\",\"S3Output\":{\"S3Uri\":\"s3://$BUCKET/$PREFIX/output\",\"LocalPath\":\"/opt/ml/processing/output\",\"S3UploadMode\":\"EndOfJob\"}}]"

echo "=== Job submitted successfully. Waiting for completion... ==="
aws sagemaker wait processing-job-completed-or-stopped --processing-job-name "$JOB_NAME"

echo "=== [4/4] Syncing Shards and Merging Final Submission ==="
mkdir -p output/sagemaker_shards
aws s3 sync "s3://$BUCKET/$PREFIX/output" output/sagemaker_shards
python sagemaker/merge_submission.py --shard-dir output/sagemaker_shards --output-dir output --test-dir student_resource/dataset/test
