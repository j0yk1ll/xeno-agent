import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from abc import ABC, abstractmethod
from typing import Optional

class StorageBackend(ABC):
    @abstractmethod
    def save_file(self, file_ref: str, data: bytes) -> None:
        pass

    @abstractmethod
    def read_file(self, file_ref: str) -> bytes:
        pass

class LocalStorageBackend(StorageBackend):
    def __init__(self, base_directory: str):
        self.base_directory = base_directory
        os.makedirs(self.base_directory, exist_ok=True)

    def save_file(self, file_ref: str, data: bytes) -> None:
        full_path = os.path.join(self.base_directory, file_ref)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            with open(full_path, 'wb') as f:
                f.write(data)
            print(f"File saved locally at {full_path}")
        except IOError as e:
            print(f"Failed to save file locally: {e}")
            raise

    def read_file(self, file_ref: str) -> bytes:
        full_path = os.path.join(self.base_directory, file_ref)
        try:
            with open(full_path, 'rb') as f:
                data = f.read()
            print(f"File read locally from {full_path}")
            return data
        except IOError as e:
            print(f"Failed to read file locally: {e}")
            raise

class S3StorageBackend(StorageBackend):
    def __init__(self, bucket_name: str, 
                 access_key_id: Optional[str] = None, 
                 secret_access_key: Optional[str] = None,
                 region: Optional[str] = None,
                 endpoint_url: Optional[str] = None):
        self.bucket_name = bucket_name
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region,
                endpoint_url=endpoint_url  # Useful for S3-compatible services
            )
            # Check if bucket exists, if not create it
            self._ensure_bucket_exists()
        except (BotoCoreError, ClientError) as e:
            print(f"Failed to initialize AWS S3 client: {e}")
            raise

    def _ensure_bucket_exists(self):
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"Bucket '{self.bucket_name}' exists.")
        except ClientError:
            # If bucket does not exist, create it
            try:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created.")
            except (BotoCoreError, ClientError) as e:
                print(f"Failed to create bucket '{self.bucket_name}': {e}")
                raise

    def save_file(self, file_ref: str, data: bytes) -> None:
        try:
            self.s3_client.put_object(Bucket=self.bucket_name, Key=file_ref, Body=data)
            print(f"File saved to AWS S3 bucket '{self.bucket_name}' at '{file_ref}'")
        except (BotoCoreError, ClientError) as e:
            print(f"Failed to save file to AWS S3: {e}")
            raise

    def read_file(self, file_ref: str) -> bytes:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_ref)
            data = response['Body'].read()
            print(f"File read from AWS S3 bucket '{self.bucket_name}' at '{file_ref}'")
            return data
        except (BotoCoreError, ClientError) as e:
            print(f"Failed to read file from AWS S3: {e}")
            raise