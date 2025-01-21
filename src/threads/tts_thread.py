import logging
import queue
import threading
from typing import Optional

from src.utils.tts import TTS
from src.utils.audio_player import AudioPlayer
from src.utils.settings_manager import SettingsManager


class TTSThread:
    """
    Class that handles Text-to-Speech (TTS) processing in a separate thread.
    """

    def __init__(self, inbound_queue: queue.Queue, settings_manager: SettingsManager):
        """
        Initializes the TTSThread.

        Args:
            inbound_queue (queue.Queue): Queue from which to receive text inputs.
            settings_manager (SettingsManager): Manager for application settings.
        """
        self.inbound_queue = inbound_queue
        self.settings_manager = settings_manager
        self.shutdown_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        settings = self.settings_manager.get_settings()
        self.tts = TTS(**settings)
        self.audio_player = AudioPlayer(**settings)

        # Register the settings update callback
        self.settings_manager.on_update(self._update_settings)

    def _update_settings(self):
        """
        Updates TTS and AudioPlayer settings when settings_manager notifies of changes.
        """
        settings = self.settings_manager.get_settings()
        if self.tts:
            self.tts.update_voice(settings.get("voice"))
            self.tts.update_desired_sample_rate(settings.get("desired_sample_rate"))
        logging.debug("TTSThread settings updated.")

    def start(self):
        """
        Starts the TTS processing thread.
        """
        if self.thread and self.thread.is_alive():
            logging.warning("TTSThread is already running.")
            return

        self.shutdown_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="TTSThread")
        self.thread.start()
        logging.info("TTSThread started.")

    def stop(self):
        """
        Signals the thread to shut down and waits for it to finish.
        """
        if not self.thread:
            logging.warning("TTSThread is not running.")
            return

        logging.info("Stopping TTSThread.")
        self.shutdown_event.set()
        # Put sentinel to unblock queue.get
        self.inbound_queue.put(None)
        self.thread.join()
        self.thread = None
        logging.info("TTSThread stopped.")

    def _run(self):
        """
        The main loop that processes text inputs and generates speech.
        """
        logging.info("TTSThread running.")

        try:
            while not self.shutdown_event.is_set():
                try:
                    # Get a message from the inbound_queue with a timeout
                    text = self.inbound_queue.get(timeout=0.05)  # 50ms timeout

                    if text is None:
                        logging.info("TTSThread received shutdown signal.")
                        break  # Sentinel received, exit the loop

                    logging.info(f"TTSThread processing text for TTS: {text}")

                    try:
                        # Generate PCM chunks and play them
                        for pcm_chunk in self.tts.generate_audio(text):
                            self.audio_player.play_audio_chunk(pcm_chunk)
                    except Exception as e:
                        logging.error(f"Error during TTS processing: {e}", exc_info=True)
                    finally:
                        self.inbound_queue.task_done()

                except queue.Empty:
                    continue  # No message received, continue the loop

        except Exception as e:
            logging.error(f"Exception in TTSThread: {e}", exc_info=True)

        finally:
            # Clean up resources
            if self.audio_player:
                self.audio_player.close()
            logging.info("TTSThread shutting down.")
