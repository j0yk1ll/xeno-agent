import logging
import queue
import threading

from src.utils.tts import TTS
from src.utils.audio_player import AudioPlayer
from src.utils.settings_manager import SettingsManager


def tts_thread(
    inbound_queue: queue.Queue,
    shutdown_event: threading.Event,
    settings_manager: SettingsManager,
):
    """
    Runs the TTS processing loop in a separate thread, communicating via queue.Queue.

    Args:
        inbound_queue (queue.Queue): Queue from which to receive text inputs.
        stop_event (threading.Event): Event to signal the thread to stop.
        voice (str, optional): Voice identifier for TTS. Defaults to "af_sky".
        desired_sample_rate (int, optional): Sample rate for audio playback. Defaults to 24000.
    """

    def _update_settings():
        tts.update_voice(settings["voice"])
        tts.update_desired_sample_rate(settings["desired_sample_rate"])

    settings = settings_manager.get_settings()

    settings_manager.on_update(_update_settings)

    # Initialize the TTS engine
    tts = TTS(**settings)

    # Initialize the AudioPlayer
    audio_player = AudioPlayer(**settings)

    try:
        while not shutdown_event.is_set():
            try:
                # Attempt to get a message from the inbound_queue with a timeout
                text = inbound_queue.get(timeout=0.05)  # 50ms timeout

                if text is None:
                    logging.info("Received shutdown signal.")
                    break  # Sentinel received, exit the loop

                logging.info(f"Processing text for TTS: {text}")

                try:
                    # Generate PCM chunks and play them
                    for pcm_chunk in tts.generate_audio(text):
                        audio_player.play_audio_chunk(pcm_chunk)
                except Exception as e:
                    logging.error(f"Error during TTS processing: {e}")
                finally:
                    inbound_queue.task_done()

            except queue.Empty:
                continue  # No message received, continue the loop

    except Exception as e:
        logging.error(f"Exception in TTS thread: {e}")
    finally:
        # Clean up resources
        audio_player.close()
        logging.info("TTS Thread shutting down.")
