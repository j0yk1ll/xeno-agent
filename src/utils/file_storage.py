

from src.utils.file_storage_backends import LocalStorageBackend, S3StorageBackend, StorageBackend


class FileStorage:
    def __init__(self, backend: StorageBackend):
        """
        Initialize the FileStorage with a StorageBackend instance.

        :param backend: An instance of a StorageBackend subclass
        """
        self.backend = backend

    def save_file(self, file_ref: str, data: bytes) -> None:
        """
        Save a file to the configured storage backend.

        :param file_ref: File reference
        :param data: File data in bytes
        """
        self.backend.save_file(file_ref, data)

    def read_file(self, file_ref: str) -> bytes:
        """
        Read a file from the configured storage backend.

        :param file_ref: File reference
        :return: File data in bytes
        """
        return self.backend.read_file(file_ref)

# Example Usage
if __name__ == "__main__":
    from io import BytesIO
    from PIL import Image

    # Generate a dummy image
    image = Image.new('RGB', (100, 100), color=(255, 0, 0))  # Create a red 100x100 image
    image_buffer = BytesIO()
    image.save(image_buffer, format='PNG')
    file_data = image_buffer.getvalue()  # Get binary image data
    file_ref = "dummy_image.png"

    # Use local filesystem
    local_backend = LocalStorageBackend(base_directory='./local_data')
    local_storage = FileStorage(backend=local_backend)
    local_storage.save_file(file_ref, file_data)
    read_data = local_storage.read_file(file_ref)
    print(f"Read from local: {read_data.decode()}")

    # Use AWS S3
    aws_backend = S3StorageBackend(
        bucket_name='my-aws-bucket',
        access_key_id='YOUR_AWS_ACCESS_KEY',
        secret_access_key='YOUR_AWS_SECRET_KEY',
        region='us-west-2',
        endpoint_url='https://s3.amazonaws.com'  # AWS S3 endpoint
    )
    aws_storage = FileStorage(backend=aws_backend)
    aws_storage.save_file(file_ref, file_data)
    read_data_aws = aws_storage.read_file(file_ref)
    print(f"Read from AWS S3: {read_data_aws.decode()}")

    # Use Wasabi
    wasabi_backend = S3StorageBackend(
        bucket_name='my-wasabi-bucket',
        access_key_id='YOUR_WASABI_ACCESS_KEY',
        secret_access_key='YOUR_WASABI_SECRET_KEY',
        region='us-east-1',
        endpoint_url='https://s3.wasabisys.com'
    )
    wasabi_storage = FileStorage(backend=wasabi_backend)
    wasabi_storage.save_file(file_ref, file_data)
    read_data_wasabi = wasabi_storage.read_file(file_ref)
    print(f"Read from Wasabi: {read_data_wasabi.decode()}")
