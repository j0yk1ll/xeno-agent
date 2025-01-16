import logging
import queue
import threading
from typing import Optional

import sounddevice as sd

# Constants
DEFAULT_CHUNK_SIZE = 512  # 512 samples at 16 kHz
DEFAULT_SAMPLE_RATE = 16000

class AudioRecorder:
    """
    Handles audio recording from the microphone using sounddevice.
    Audio chunks are enqueued into a thread-safe queue for processing.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        queue_maxsize: int = 100,
    ):
        """
        Initializes the AudioRecorder.

        :param sample_rate: Audio sample rate.
        :param chunk_size: Number of samples per audio chunk.
        :param queue_maxsize: Maximum number of chunks to hold in the queue.
        """
        self.logger = logging.getLogger(__name__)

        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue(maxsize=queue_maxsize)
        self._stream: Optional[sd.RawInputStream] = None
        self.shutdown_event = threading.Event()

    def start_recording(self):
        """
        Start the microphone stream (non-blocking).
        """
        self.logger.info("Starting AudioRecorder...")
        self.shutdown_event.clear()

        if self._stream is not None:
            self.logger.warning("AudioRecorder stream is already running. Skipping...")
            return

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            self._stream.start()
            self.logger.info("AudioRecorder started.")
        except Exception as e:
            self.logger.error(f"Failed to start AudioRecorder: {e}")
            raise

    def stop_recording(self):
        """
        Stop the microphone stream and signal shutdown.
        """
        self.logger.info("Stopping AudioRecorder...")
        self.shutdown_event.set()

        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
                self.logger.info("AudioRecorder stopped.")
            except Exception as e:
                self.logger.error(f"Error stopping AudioRecorder: {e}")
            finally:
                self._stream = None
        else:
            self.logger.warning("AudioRecorder stream was not running.")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function called by sounddevice for each audio block.
        Enqueues audio data into the audio_queue.
        """
        if status:
            self.logger.warning(f"AudioRecorder callback status: {status}")

        if self.shutdown_event.is_set():
            return

        try:
            self.audio_queue.put(bytes(indata), timeout=0.1)
            self.logger.debug(
                f"Audio chunk enqueued. Queue size: {self.audio_queue.qsize()}"
            )
        except queue.Full:
            self.logger.warning("AudioRecorder queue is full. Dropping audio chunk.")
