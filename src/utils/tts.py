import os
import re
import logging
from typing import Iterator
from pathlib import Path

import num2words
import numpy as np
import emoji
import torch
import spacy

from src.utils.audio_player import AudioPlayer

from .kokoro.core import generate
from .kokoro.models import build_model


# Define the xeno directory
xeno_models_dir = Path.home() / ".xeno" / "models"
xeno_spacy_models_dir = xeno_models_dir / "spacy"
xeno_kokoro_models_dir = xeno_models_dir / "kokoro"


def get_available_voices() -> list:
    """
    Scans the 'kokoro/voices' directory and returns a list of available voice names.

    Returns:
        List[str]: A list of available voice names.

    Raises:current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        voicepack_path = os.path.join(
            current_directory, "kokoro", "voices", f"{self.voice}.pt"
        )

        if not os.path.exists(voicepack_path):
            raise FileNotFoundError(
                f"Voicepack '{self.voice}' does not exist in 'kokoro/voices' directory."
            )
        FileNotFoundError: If the voices directory does not exist.
    """
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    voices_dir = os.path.join(current_directory, "kokoro", "voices")

    if not os.path.isdir(voices_dir):
        raise FileNotFoundError(f"Voices directory '{voices_dir}' does not exist.")

    voices = [file[:-3] for file in os.listdir(voices_dir) if file.endswith(".pt")]
    return voices


class TTS:
    def __init__(self, desired_sample_rate=24000, voice="af_sky", **kwargs):
        """
        Initialize the TTS engine:
          - Loads spaCy (English)
          - Loads the Kokoro TTS model
          - Initializes voicepacks dictionary (voicepacks are loaded on-demand)
          - Sets a desired sample rate (does not configure audio devices).
        """
        self.model = None
        self.voicepack = None
        self.desired_sample_rate = desired_sample_rate
        self.voice = voice

        # We lock access to the TTS model so only one call at a time
        # can safely execute the generate() function (if needed).
        self.model_lock = (
            None  # You can leave this as None or use e.g. threading.Lock() externally
        )

        self._initialize_spacy()
        self._initialize_model()
        self._load_voice()

    def _initialize_spacy(self):
        """
        Load the 'en_core_web_sm' spaCy model for sentence parsing.

        Raises:
            OSError: If the spaCy model cannot be loaded.
        """
        try:
            self.spacy_nlp = spacy.load(xeno_spacy_models_dir)
        except OSError:
            raise OSError(
                "SpaCy model 'en_core_web_sm' is not available. Please ensure it is downloaded."
            )

    def _initialize_model(self):
        """
        Loads the Kokoro model.

        Raises:
            FileNotFoundError: If the Kokoro model file does not exist.
        """
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")

        # Load the model
        model_path = xeno_kokoro_models_dir / "kokoro-v0_19.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro model path '{model_path}' does not exist.")
        self.model = build_model(model_path, device)
        logging.info("Kokoro model loaded successfully.")

    def _load_voice(self):
        """
        Loads the specific voicepack.

        Raises:
            FileNotFoundError: If the specified voicepack does not exist.
            Exception: If loading the voicepack fails.
        """

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        voicepack_path = os.path.join(
            current_directory, "kokoro", "voices", f"{self.voice}.pt"
        )

        if not os.path.exists(voicepack_path):
            raise FileNotFoundError(
                f"Voicepack '{self.voice}' does not exist in 'kokoro/voices' directory."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            voicepack = torch.load(
                voicepack_path, map_location=device, weights_only=True
            )
            self.voicepack = voicepack.to(device)
            logging.info(f"Loaded voice: {self.voice}")
        except Exception as e:
            logging.error(f"Failed to load voice '{self.voice}': {e}")
            raise e

    def _clean_sentence(self, text: str) -> str:
        """
        Basic text cleanup and normalization:
          - Replace newlines, smart quotes, apostrophes, dashes
          - Convert numeric tokens to words
          - Remove or replace certain punctuation/symbols
        """
        # 1) Replace newlines with space
        text = text.replace("\n", " ")

        # 2) Replace smart quotes and apostrophes
        text = (
            text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        )

        # 3) Replace longer dashes with hyphen
        text = re.sub(r"[–—]", "-", text)

        # 4) Symbol mapping
        symbol_mapping = {
            "**": "",
            "*": "",
            "@": " at ",
            "#": "",
            "$": " dollar ",
            "€": " euro ",
            "%": " percent ",
            "&": " and ",
            "+": " plus ",
            "=": " equals ",
            "/": " divided by ",
            "\\": " backslash ",
            "(": "",
            ")": "",
            "[": "",
            "]": "",
            "{": "",
            "}": "",
            ":": "",
            ";": "",
            "<": " less than ",
            ">": " greater than ",
            "|": " or ",
            "^": "",
            "~": "",
            "`": "",
            '"': "",
            "'": "",
        }
        for symbol, replacement in symbol_mapping.items():
            text = text.replace(symbol, replacement)

        # 5) Convert numbers to words
        def convert_numbers(txt):
            return re.sub(
                r"\b\d+(\.\d+)?\b",
                lambda match: (
                    num2words.num2words(float(match.group()))
                    if "." in match.group()
                    else num2words.num2words(int(match.group()))
                ),
                txt,
            )

        text = convert_numbers(text)

        # 6) Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _split_into_sentences(
        self,
        text: str,
        cleanup_links: bool = True,
        cleanup_emojis: bool = True,
        max_length: int = 200,
    ) -> Iterator[str]:
        """
        Splits the input text into smaller sentences (via spaCy), cleans them,
        and yields them one by one. If a sentence is too long, further chunk it.
        """
        # Remove URLs
        if cleanup_links:
            link_pattern = re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|"
                r"(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            )
            text = link_pattern.sub("", text)

        # Remove emojis
        if cleanup_emojis:
            text = emoji.replace_emoji(text, "")

        text = text.strip()
        if not text:
            return

        # Use spaCy to parse sentences
        doc = self.spacy_nlp(text)
        for spacy_sent in doc.sents:
            cleaned = self._clean_sentence(spacy_sent.text.strip())
            if not cleaned:
                continue

            # If it's short enough, yield directly;
            # otherwise chunk it further.
            if len(cleaned) <= max_length:
                yield cleaned
            else:
                # Manually chunk long sentences by punctuation/conjunction
                yield from self._split_long_sentence_spacy(spacy_sent, max_length)

    def _split_long_sentence_spacy(
        self, spacy_sentence, max_length: int
    ) -> Iterator[str]:
        """
        Splits a spacy Span (sentence) into multiple smaller chunks
        based on punctuation and coordinating conjunctions, if the
        chunk is exceeding max_length.
        """
        # Potential split points
        split_tokens = {",", ";", "and", "but", "or", "so", "yet", "for", "nor"}

        current_chunk = []
        current_length = 0

        for token in spacy_sentence:
            current_chunk.append(token.text)
            current_length += len(token.text) + 1  # +1 for space

            # If this token is a potential chunk boundary
            if token.text.lower() in split_tokens or token.dep_.lower() == "cc":
                if current_length >= max_length:
                    chunk_text = " ".join(current_chunk).strip()
                    if chunk_text:
                        yield self._clean_sentence(chunk_text)
                    current_chunk = []
                    current_length = 0

        # Any remainder
        if current_chunk:
            chunk_text = " ".join(current_chunk).strip()
            if chunk_text:
                yield self._clean_sentence(chunk_text)

    def _audio_float_to_int16_bytes(self, audio_data: np.ndarray) -> bytes:
        """
        Convert float32 audio data (-1.0..1.0) to raw 16-bit PCM data (mono).
        """
        audio_int16 = np.int16(audio_data * 32767)
        return audio_int16.tobytes()

    def _generate_audio_for_sentence(self, sentence: str) -> Iterator[bytes]:
        """
        Generates and yields small raw PCM chunks (~100ms each) for a single sentence.
        """

        # We'll break the generated audio into ~100ms chunks
        chunk_ms = 100
        chunk_samples = int(self.desired_sample_rate * (chunk_ms / 1000.0))

        # Generate the entire sentence audio in memory
        # If you do want to ensure thread safety, you can guard with self.model_lock.
        # E.g.: with self.model_lock: ...
        audio_chunks, _ = generate(
            self.model,
            sentence,
            self.voicepack,
            lang=self.voice[0],  # e.g. 'a' for "af_sky"
        )

        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks).astype(np.float32)
        else:
            combined_audio = np.array([], dtype=np.float32)

        # Normalize to avoid clipping
        max_amp = np.max(np.abs(combined_audio)) if combined_audio.size > 0 else 0.0
        if max_amp > 0:
            combined_audio /= max_amp

        # Split into smaller chunks and yield them as raw 16-bit PCM
        start = 0
        while start < len(combined_audio):
            end = start + chunk_samples
            audio_chunk = combined_audio[start:end]
            start = end
            yield self._audio_float_to_int16_bytes(audio_chunk)

    def generate_audio(self, text: str) -> Iterator[bytes]:
        """
        Splits the given text into sentences, generates audio chunks for each sentence,
        and yields them (16-bit PCM mono). This method is synchronous and does not
        handle threading or playback.
        """

        # Split text into sentences
        sentences_list = list(
            self._split_into_sentences(
                text, cleanup_links=True, cleanup_emojis=True, max_length=200
            )
        )

        for sentence in sentences_list:
            logging.info(f"Generating audio for sentence: {sentence}")
            for raw_chunk in self._generate_audio_for_sentence(sentence):
                yield raw_chunk

    def update_voice(self, voice: str):
        self.voice = voice
        self._load_voice()

    def update_desired_sample_rate(self, desired_sample_rate: int):
        self.desired_sample_rate = desired_sample_rate


if __name__ == "__main__":
    # Example usage (blocking). In production, you likely want to
    # feed the generated chunks to an audio output or queue, etc.
    logging.basicConfig(level=logging.INFO)

    tts = TTS(desired_sample_rate=24000, voice="af_sky")

    audio_player = AudioPlayer(desired_sample_rate=24000)

    sample_text = (
        "Hello world! This is a test of the stripped-down TTS. "
        "We have removed all threading and playback logic from this class."
    )

    try:
        # Generate PCM data and play each chunk
        for chunk in tts.generate_audio(sample_text):
            audio_player.play_audio_chunk(chunk)

        print("Done generating and playing audio.")
    except KeyboardInterrupt:
        print("Playback interrupted by user.")
    finally:
        # Ensure the audio stream is properly closed
        audio_player.close()

    print("Done generating audio.")
