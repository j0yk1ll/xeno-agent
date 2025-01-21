from datetime import datetime
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Optional

import mss
from PIL import Image

from src.utils.settings_manager import SettingsManager
from src.utils.llms.vision import VisionLLM


class ScreenCaptureThread:
    """
    Class that captures desktop screenshots at regular intervals in a separate thread.
    Supports Linux, Windows, and macOS platforms.
    """

    def __init__(
        self,
        outbound_queue: queue.Queue,
        settings_manager: SettingsManager,
        capture_interval: float = 1,  # Interval in seconds between captures
        queue_maxsize: int = 100,
    ):
        """
        Initializes the ScreenCaptureThread.

        Args:
            outbound_queue (queue.Queue): Queue to place captured screenshots.
            capture_interval (float, optional): Time interval between captures in seconds. Defaults to 5.0.
            queue_maxsize (int, optional): Maximum size of the outbound queue. Defaults to 100.
        """
        self.capture_interval = capture_interval
        self.queue_maxsize = queue_maxsize

        self.outbound_queue = outbound_queue
        self.is_active_event = threading.Event()
        self.shutdown_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.settings_manager = settings_manager
        settings = settings_manager.get_settings()
        
        self.vision_llm = VisionLLM(
            model_id=settings['vision_model_id'],
            api_base=settings['vision_api_base'],
            api_key=settings['vision_api_key'],
            requests_per_minute=5,
        )

        # Register the settings update callback
        self.settings_manager.on_update(self._update_settings)

    def _update_settings(self):
        """
        Updates the MemoryAgent's models based on the latest settings.
        """
        settings = self.settings_manager.get_settings()
        self.vision_llm = VisionLLM(
            model_id=settings['vision_model_id'],
            api_base=settings['vision_api_base'],
            api_key=settings['vision_api_key'],
            requests_per_minute=5,
        )
        logging.debug("ScreenCaptureThread settings updated.")

    def start(self):
        """
        Starts the screen capture thread.
        """
        if self.thread and self.thread.is_alive():
            logging.warning("ScreenCaptureThread is already running.")
            return

        self.shutdown_event.clear()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="ScreenCaptureThread"
        )
        self.thread.start()
        logging.info("ScreenCaptureThread started.")

    def stop(self):
        """
        Signals the thread to stop and waits for it to finish.
        """
        if not self.thread:
            logging.warning("ScreenCaptureThread is not running.")
            return

        self.shutdown_event.set()
        self.thread.join()
        self.thread = None
        logging.info("ScreenCaptureThread stopped.")

    def _run(self):
        """
        The main loop that captures screenshots at regular intervals and places them into the outbound queue.
        """
        logging.info("ScreenCaptureThread running.")

        # Instantiate mss within the thread
        with mss.mss() as mss_instance:
            try:
                while not self.shutdown_event.is_set():
                    if self.is_active_event.is_set():
                        start_time = time.time()
                        try:
                            # Capture the screen
                            screenshot = mss_instance.grab(mss_instance.monitors[0])  # Capture the primary monitor
                            image = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)

                            # Save the image to ~/.xeno/screenshots
                            try:
                                # Define the directory path
                                save_dir = Path.home() / '.xeno' / 'screenshots'
                                # Create the directory if it doesn't exist
                                save_dir.mkdir(parents=True, exist_ok=True)

                                # Generate a unique filename using the current timestamp
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                filename = f'screenshot_{timestamp}.png'
                                file_path = save_dir / filename

                                # Save the image
                                image.save(file_path, format='PNG')
                                logging.debug(f"Screenshot saved to {file_path}")
                            except Exception as e:
                                logging.warning(f"Failed to save screenshot: {e}")

                            try:
                                description = self.vision_llm.call(image=image)
                            except Exception as e:
                                logging.warning(
                                    f"An error occurred during screen capture: {str(e)}"
                                )
                                continue

                            message = {
                                "description": description,
                                "image": image,
                            }

                            # Put the image into the outbound queue
                            if self.outbound_queue:
                                try:
                                    self.outbound_queue.put(message, timeout=0.1)
                                    logging.debug(
                                        f"Screenshot captured and placed into outbound queue."
                                    )
                                except queue.Full:
                                    logging.warning(
                                        "Outbound queue is full. Dropping screenshot."
                                    )
                            else:
                                logging.warning("Outbound queue is not set.")
                        except Exception as e:
                            logging.error(f"Error capturing screenshot: {e}", exc_info=True)

                        # Calculate the time to sleep to maintain the capture interval
                        elapsed_time = time.time() - start_time
                        sleep_time = max(0, self.capture_interval - elapsed_time)
                        time.sleep(sleep_time)
                        continue

                    # Prevent tight loop
                    time.sleep(0.01)
            except Exception as e:
                logging.error(f"Exception in ScreenCaptureThread: {e}", exc_info=True)
            finally:
                logging.info("ScreenCaptureThread shutting down.")
