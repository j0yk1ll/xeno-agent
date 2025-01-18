import logging
import urllib.request
from pathlib import Path
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_file(url, destination):
    """
    Downloads a file from a URL to a specified destination.
    """
    try:
        logging.info(f"Downloading model from {url} to {destination}...")
        urllib.request.urlretrieve(url, destination)
        logging.info("Download completed successfully.")
    except Exception as e:
        logging.error(f"Failed to download the file. Error: {e}")
        raise

def unzip_file(zip_path, extract_to):
    """
    Unzips a file to a specified destination directory.
    """
    try:
        logging.info(f"Unzipping file {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logging.info("Unzipping completed successfully.")
    except Exception as e:
        logging.error(f"Failed to unzip the file. Error: {e}")
        raise

# Directories and URLs
xeno_models_dir = Path.home() / ".xeno" / "models"
xeno_models_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

zip_file_path = xeno_models_dir / "models.zip"
model_files_url = "https://huggingface.co/j0yk1ll/xeno-agent-model-files/resolve/main/models.zip?download=true"

# Download and unzip the file
download_file(model_files_url, zip_file_path)
unzip_file(zip_file_path, xeno_models_dir)

# Delete the zip file after extraction
try:
    zip_file_path.unlink()
    logging.info(f"Deleted zip file: {zip_file_path}")
except Exception as e:
    logging.warning(f"Failed to delete zip file: {zip_file_path}. Error: {e}")
