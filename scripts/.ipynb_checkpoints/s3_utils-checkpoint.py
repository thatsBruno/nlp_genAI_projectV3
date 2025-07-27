import os
import boto3


def upload_to_s3(bucket_name: str, local_path: str, s3_prefix: str) -> str:
    """Upload local file or folder to S3 and return base S3 URI."""
    s3_client = boto3.client('s3')

    if os.path.isfile(local_path):
        # Single file upload
        s3_path = f"{s3_prefix}/{os.path.basename(local_path)}"
        s3_client.upload_file(local_path, bucket_name, s3_path)
        return f"s3://{bucket_name}/{s3_path}"

    elif os.path.isdir(local_path):
        # Directory upload
        for root, _, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, start=local_path)
                s3_path = f"{s3_prefix}/{relative_path.replace(os.sep, '/')}"
                s3_client.upload_file(local_file_path, bucket_name, s3_path)
        return f"s3://{bucket_name}/{s3_prefix}/"

    else:
        raise ValueError("The provided local_path is neither a file nor a directory.")


if __name__ == '__main__':
    bucket_name = 's3nlpbananas'
    output_dir = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/models/multi/grupo2.tar.gz"

    upload_to_s3(bucket_name, output_dir, "models/modelo_final")
