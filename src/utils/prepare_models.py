import logging
from faster_whisper import WhisperModel
import spacy
import spacy.cli
import torch


def prepare_models():
    logging.info("Preparing Whisper transcription model ('base')...")
    WhisperModel(model_size_or_path="base", device="cpu")

    logging.info("Preparing Whisper realtime transcription model ('tiny')...")
    WhisperModel(model_size_or_path="tiny", device="cpu")

    logging.info("Preparing Silero VAD model ('snakers4/silero-vad')...")
    torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )

    logging.info("Preparing SpaCy language model ('en_core_web_sm')...")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")

    print("All models downloaded successfully.")


if __name__ == "__main__":
    import logging
    prepare_models()
