import boto3
import pandas as pd
import logging
from src.utils.logger import logger
from io import StringIO


class s3_operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):
        """
        Initialize the s3_operations class with AWS credentials and S3 bucket details.
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info("Data Ingestion from S3 bucket initialized")

    def fetch_file_from_s3(self, file_key):
        """
        Fetches a CSV file from the S3 bucket and returns it as a Pandas DataFrame.
        :param file_key: S3 file path (e.g., 'data/data.csv')
        :return: Pandas DataFrame
        """
        try:
            logging.info(f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logging.info(f"Successfully fetched and loaded '{file_key}' from S3 that has {len(df)} records.")
            return df
        except Exception as e:
            logging.exception(f"❌ Failed to fetch '{file_key}' from S3: {e}")
            return None

    def upload_file(self, local_path, s3_key):
        """
        Uploads a local file to the S3 bucket.
        :param local_path: Local file path
        :param s3_key: S3 destination key (e.g., 'production/model.pkl')
        """
        try:
            logging.info(f"Uploading '{local_path}' → s3://{self.bucket_name}/{s3_key} ...")
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logging.info(f"✅ Upload successful: s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            logging.exception(f"❌ Failed to upload '{local_path}' to S3: {e}")
            raise