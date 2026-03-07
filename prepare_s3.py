import boto3

bucket_name = "sagemaker-studio-970361933570-9nh5vyj1jvq"

folders = [
    "datasets/titanic/raw/",
    "datasets/titanic/extracted/",
    "datasets/titanic/features/",
    
    "pipelines/titanic/processing/",
    "pipelines/titanic/training/",
    "pipelines/titanic/evaluation/",
    
    "models/titanic/",
    
    "batch-inference/titanic/",
    
    "logs/titanic/"
]

s3 = boto3.client("s3")

for folder in folders:
    s3.put_object(
        Bucket=bucket_name,
        Key=f"{folder}.keep"
    )
    print(f"Created {folder}")

print("S3 structure initialized.")