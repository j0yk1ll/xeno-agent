import logging
import queue
import threading
import time
from typing import Optional
from src.utils.audio_recorder import AudioRecorder
from src.utils.stt import STT

class STTThread:
    """
    Class that handles Speech-to-Text (STT) processing in a separate thread.
    """
    def __init__(
        self,
        outbound_queue: queue.Queue,
        vad_threshold: float = 0.75,
        sample_rate: int = 16000,
        silence_duration: float = 2.0,
        pre_speech_length: float = 0.5,
        chunk_size: int = 512,
        queue_maxsize: int = 100,
    ):
        self.vad_threshold = vad_threshold
        self.sample_rate = sample_rate
        self.silence_duration = silence_duration
        self.pre_speech_length = pre_speech_length
        self.chunk_size = chunk_size
        self.queue_maxsize = queue_maxsize

        self.outbound_queue = outbound_queue
        self.is_active_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.stt: Optional[STT] = None
        self.audio_recorder: Optional[AudioRecorder] = None

    def start(self):
        if self.thread and self.thread.is_alive():
            logging.warning("STTThread is already running.")
            return

        self.shutdown_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="STTThread")
        self.thread.start()
        logging.info("STTThread started.")

    def stop(self):
        if not self.thread:
            logging.warning("STTThread is not running.")
            return

        self.shutdown_event.set()
        self.thread.join()
        self.thread = None
        logging.info("STTThread stopped.")

    def _on_complete_transcription(self, text: str):
        if self.outbound_queue:
            try:
                self.outbound_queue.put(text, timeout=0.1)
                logging.debug(f"Transcription sent to outbound queue: {text}")
            except queue.Full:
                logging.warning("Outbound queue is full. Dropping transcription.")
        else:
            logging.warning("Outbound queue is not set.")

    def _run(self):
        logging.info("STTThread running.")

        # Initialize AudioRecorder
        self.audio_recorder = AudioRecorder(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            queue_maxsize=self.queue_maxsize,
        )

        # Initialize STT
        try:
            self.stt = STT(
                vad_threshold=self.vad_threshold,
                sample_rate=self.sample_rate,
                silence_duration=self.silence_duration,
                pre_speech_length=self.pre_speech_length,
                on_complete_transcription=self._on_complete_transcription,
            )
        except Exception as e:
            logging.error(f"Failed to initialize STT: {e}")
            return

        is_recording = False

        try:
            while not self.shutdown_event.is_set():
                # Manage recording state
                if self.is_active_event.is_set() and not is_recording:
                    self.audio_recorder.start_recording()
                    is_recording = True
                    logging.info("Recording started.")
                elif not self.is_active_event.is_set() and is_recording:
                    self.audio_recorder.stop_recording()
                    is_recording = False
                    logging.info("Recording stopped.")

                # Process audio data
                try:
                    audio_data = self.audio_recorder.audio_queue.get(timeout=0.1)  # 100ms timeout
                    logging.debug(f"Processing audio data of length {len(audio_data)} bytes.")
                    self.stt.process_audio(audio_data)
                    self.audio_recorder.audio_queue.task_done()
                except queue.Empty:
                    pass  # No audio data received
                except Exception as e:
                    logging.error(f"Error processing audio data: {e}", exc_info=True)

                # Prevent tight loop
                time.sleep(0.01)

        except Exception as e:
            logging.error(f"Exception in STTThread: {e}", exc_info=True)

        finally:
            # Ensure recording is stopped
            if is_recording:
                self.audio_recorder.stop_recording()
                logging.info("Recording stopped during shutdown.")

            logging.info("STTThread shutting down.")
