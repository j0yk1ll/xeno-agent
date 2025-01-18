import logging
import urllib.request
from pathlib import Path
import faster_whisper
import spacy
import spacy.cli
import torch

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

def prepare_models():
    # Define the xeno directory
    xeno_models_dir = Path.home() / ".xeno" / "models"
    xeno_models_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    xeno_faster_whisper_models_dir = xeno_models_dir / "faster_whisper"
    xeno_faster_whisper_models_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    xeno_silero_vad_models_dir = xeno_models_dir / "silero_vad"
    xeno_silero_vad_models_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    xeno_spacy_models_dir = xeno_models_dir / "spacy"
    xeno_spacy_models_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    xeno_audioclip_models_dir = xeno_models_dir / "audioclip"
    xeno_audioclip_models_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    xeno_kokoro_models_dir = xeno_models_dir / "kokoro"
    xeno_kokoro_models_dir.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

    # Prepare Whisper transcription model ('base')
    logging.info("Preparing Whisper transcription model ('base')...")
    faster_whisper.download_model("base", output_dir=xeno_faster_whisper_models_dir)

    # Prepare Silero VAD model ('snakers4/silero-vad')
    logging.info("Preparing Silero VAD model ('snakers4/silero-vad')...")
    silero_vad_url =  "https://raw.githubusercontent.com/snakers4/silero-vad/9060f664f20eabb66328e4002a41479ff288f14c/src/silero_vad/data/silero_vad.jit"
    silero_vad_path = xeno_silero_vad_models_dir / "silero_vad.jit"

    if not silero_vad_path.exists():
        logging.info(f"Model not found at {silero_vad_path}. Initiating download...")
        download_file(silero_vad_url, silero_vad_path)
    else:
        logging.info(f"Model already exists at {silero_vad_path}. Skipping download.")

    # Prepare SpaCy language model ('en_core_web_sm')
    logging.info("Preparing SpaCy language model ('en_core_web_sm')...")
    if not xeno_spacy_models_dir.exists():
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        nlp.to_disk(xeno_spacy_models_dir)

    # Prepare AudioCLIP models
    audio_clip_url = "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/AudioCLIP-Full-Training.pt"
    clip_url = "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/CLIP.pt"
    esrnxfbsp_url = "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/ESRNXFBSP.pt"
    bpe_url = "https://github.com/AndreyGuzhov/AudioCLIP/releases/download/v0.1/bpe_simple_vocab_16e6.txt.gz"

    audio_clip_path = xeno_audioclip_models_dir / "AudioCLIP-Full-Training.pt"
    clip_path = xeno_audioclip_models_dir / "CLIP.pt"
    esrnxfbsp_path = xeno_audioclip_models_dir / "ESRNXFBSP.pt"
    bpe_path = xeno_audioclip_models_dir / "bpe_simple_vocab_16e6.txt.gz"

    for url, path in [
        (audio_clip_url, audio_clip_path),
        (clip_url, clip_path),
        (esrnxfbsp_url, esrnxfbsp_path),
        (bpe_url, bpe_path),
    ]:
        if not path.exists():
            logging.info(f"Model not found at {path}. Initiating download...")
            download_file(url, path)
        else:
            logging.info(f"Model already exists at {path}. Skipping download.")

    # Prepare Kokoro model
    kokoro_url = "https://huggingface.co/geneing/Kokoro/resolve/f610f07c62f8baa30d4ed731530e490230e4ee83/kokoro-v0_19.pth"
    kokoro_path = xeno_kokoro_models_dir / "kokoro-v0_19.pth"

    if not kokoro_path.exists():
        logging.info(f"Kokoro model not found at {kokoro_path}. Initiating download...")
        download_file(kokoro_url, kokoro_path)
    else:
        logging.info(f"Kokoro model already exists at {kokoro_path}. Skipping download.")

    logging.info("All models are prepared successfully.")

if __name__ == "__main__":
    prepare_models()
