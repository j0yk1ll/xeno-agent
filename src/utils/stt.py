import logging
import time
import re
from typing import Optional, Callable

import numpy as np
import torch
from faster_whisper import WhisperModel

# Constants
INT16_MAX = 32767
DEFAULT_CHUNK_SIZE = 512  # 512 samples at 16 kHz
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_SILENCE_DURATION = 2.0
DEFAULT_VAD_THRESHOLD = 0.75

class STT:
    """
    Manages speech-to-text (STT) using Whisper and Silero VAD models.
    Processes audio data ingested via the `process_audio` method.
    """

    def __init__(
        self,
        transcription_model: str = "base",
        vad_model_repo: str = "snakers4/silero-vad",
        vad_threshold: float = DEFAULT_VAD_THRESHOLD,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        silence_duration: float = DEFAULT_SILENCE_DURATION,
        pre_speech_length: float = 0.5,
        on_complete_transcription: Optional[Callable[[str], None]] = None,
    ):
        """
        Initializes the STT system with specified parameters.

        :param transcription_model: Name or path of the Whisper model (e.g., 'base').
        :param vad_model_repo: HuggingFace repo for Silero VAD (e.g., 'snakers4/silero-vad').
        :param vad_threshold: Probability threshold above which a chunk is considered speech.
        :param sample_rate: Audio sample rate.
        :param silence_duration: Seconds of silence before finalizing transcription.
        :param pre_speech_length: Seconds of audio to buffer before speech starts.
        :param on_complete_transcription: Callback invoked with complete transcription text.
        """

        logging.debug("Initializing STT with parameters:")
        logging.debug(f"  transcription_model: {transcription_model}")
        logging.debug(f"  vad_model_repo: {vad_model_repo}")
        logging.debug(f"  vad_threshold: {vad_threshold}")
        logging.debug(f"  sample_rate: {sample_rate}")
        logging.debug(f"  silence_duration: {silence_duration}")
        logging.debug(f"  pre_speech_length: {pre_speech_length}")

        # Assign parameters
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.pre_speech_length = pre_speech_length
        self.vad_threshold = vad_threshold
        self.on_complete_transcription = on_complete_transcription

        # Silero VAD
        try:
            self.vad_model, _ = torch.hub.load(
                repo_or_dir=vad_model_repo, model="silero_vad", force_reload=False
            )
            self.vad_model.reset_states()
            logging.debug("Silero VAD model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load Silero VAD model: {e}")
            raise

        # Whisper (for final transcription)
        try:
            self.transcription_model = WhisperModel(
                model_size_or_path=transcription_model,
                device="cpu",
            )
            logging.debug(
                f"Whisper transcription model '{transcription_model}' loaded successfully."
            )
        except Exception as e:
            logging.error(
                f"Failed to load Whisper transcription model '{transcription_model}': {e}"
            )
            raise

        # Audio state
        self.is_talking = False
        self.silence_start_time: Optional[float] = None
        self.buffered_audio = np.array([], dtype=np.float32)

        # Pre-speech buffer
        self.max_pre_speech_samples = int(pre_speech_length * sample_rate)
        self.pre_speech_buffer = np.array([], dtype=np.float32)

        # Buffer to accumulate raw bytes for exactly 512-sample frames
        self.vad_buffer = b""

    def process_audio(self, data: bytes):
        """
        Ingests and processes audio data for speech detection and transcription.

        :param data: Raw audio bytes (int16, mono).
        """
        try:
            # Accumulate partial data
            self.vad_buffer += data
            CHUNK_SIZE_BYTES = DEFAULT_CHUNK_SIZE * 2  # 512 samples * 2 bytes per sample

            # Process all complete chunks in the buffer
            while len(self.vad_buffer) >= CHUNK_SIZE_BYTES:
                frame = self.vad_buffer[:CHUNK_SIZE_BYTES]
                self.vad_buffer = self.vad_buffer[CHUNK_SIZE_BYTES:]

                # Convert int16 -> float32
                chunk_i16 = np.frombuffer(frame, dtype=np.int16)
                chunk_f32 = chunk_i16.astype(np.float32) / INT16_MAX

                # Update pre-speech buffer
                self._update_pre_speech_buffer(chunk_f32)

                # Check speech probability
                vad_prob = self._run_silero_vad(chunk_f32)
                is_speech = vad_prob > self.vad_threshold

                logging.debug(
                    f"VAD prob: {vad_prob:.4f}, is_speech: {is_speech}, "
                    f"is_talking: {self.is_talking}"
                )

                if is_speech:
                    if not self.is_talking:
                        # User started talking
                        self.is_talking = True
                        self.silence_start_time = None
                        self.buffered_audio = np.concatenate(
                            [self.pre_speech_buffer, chunk_f32]
                        )
                        logging.info("🗣 User started talking")
                    else:
                        # User continues talking
                        self.buffered_audio = np.concatenate(
                            [self.buffered_audio, chunk_f32]
                        )
                        logging.debug(
                            f"Accumulated buffered_audio length: {len(self.buffered_audio)}"
                        )
                        if self.silence_start_time is not None:
                            silence_time = time.time() - self.silence_start_time
                            if silence_time >= 1.5:
                                logging.debug(
                                    f"[User paused for {silence_time:.2f}s but resumed. "
                                    f"Resetting silence_start_time.]"
                                )
                            self.silence_start_time = None
                else:
                    if self.is_talking:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                            logging.debug("Silence detected. Starting silence timer.")

                        silence_time = time.time() - self.silence_start_time
                        logging.debug(
                            f"Silence time: {silence_time:.2f}s, threshold: {self.silence_duration}s"
                        )

                        if silence_time >= self.silence_duration:
                            # Finalize transcription
                            final_text = self._transcribe_audio(self.buffered_audio)
                            if final_text and self.on_complete_transcription:
                                logging.info(f"🗣 Full Transcription: {final_text}")
                                self.on_complete_transcription(final_text)

                            # Reset state
                            self.is_talking = False
                            self.buffered_audio = np.array([], dtype=np.float32)
                            self.silence_start_time = None
                            logging.info("🗣 User stopped talking")

        except Exception as e:
            logging.error(f"Error in STT process_audio: {e}", exc_info=True)

    def _update_pre_speech_buffer(self, chunk_f32: np.ndarray):
        """
        Maintains a buffer of audio before the speech starts.
        """
        try:
            new_buffer = np.concatenate([self.pre_speech_buffer, chunk_f32])
            if len(new_buffer) > self.max_pre_speech_samples:
                excess = len(new_buffer) - self.max_pre_speech_samples
                new_buffer = new_buffer[excess:]
                logging.debug(
                    f"Pre-speech buffer exceeded max. Truncated by {excess} samples."
                )
            self.pre_speech_buffer = new_buffer
        except Exception as e:
            logging.error(f"Failed to update pre-speech buffer: {e}")

    def _run_silero_vad(self, audio_f32: np.ndarray) -> float:
        """
        Runs the Silero VAD model on a single audio chunk.
        Returns the speech probability between 0.0 and 1.0.
        """
        try:
            with torch.inference_mode():
                prob = self.vad_model(
                    torch.from_numpy(audio_f32), self.sample_rate
                ).item()
            return prob
        except Exception as e:
            logging.error(f"Failed to run Silero VAD: {e}")
            return 0.0  # Assume no speech on error

    def _transcribe_audio(self, audio_f32: np.ndarray) -> str:
        """
        Transcribes the buffered audio using the Whisper model.
        Returns the transcribed text.
        """
        if len(audio_f32) < 2000:
            logging.debug("Audio too short for transcription. Skipping.")
            return ""

        try:
            segments, _info = self.transcription_model.transcribe(
                audio_f32, beam_size=1, language="en"
            )
            text = " ".join(seg.text for seg in segments).strip()
            text = re.sub(r"\s+", " ", text)
            logging.debug(f"Transcription result: {text}")
            return text
        except Exception as e:
            logging.error(f"Failed to transcribe audio: {e}")
            return ""
