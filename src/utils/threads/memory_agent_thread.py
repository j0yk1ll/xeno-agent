import logging
import queue
import threading
from typing import Dict, List, Optional

import PIL

from src.memory_agent.agent import MemoryAgent
from src.utils.settings_manager import SettingsManager
from src.utils.memory_manager import MemoryManager


class MemoryAgentThread:
    """
    Class that manages the MemoryAgent in a separate thread.
    """
    def __init__(
        self,
        settings_manager: SettingsManager,
        memory_manager: MemoryManager,
    ):
        """
        Initializes the MemoryAgentThread.

        Args:
            settings_manager (SettingsManager): Manager for application settings.
            memory_manager (MemoryManager): Manager for memory operations.
        """
        self.inbound_queue = queue.Queue()
        self.memory_manager = memory_manager
        self.settings_manager = settings_manager
        self.shutdown_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.memory_agent: Optional[MemoryAgent] = None
        self._setup()

    def _setup(self):
        """
        Sets up the MemoryAgent and registers the settings update callback.
        """
        settings = self.settings_manager.get_settings()

        # Initialize the MemoryAgent
        self.memory_agent = MemoryAgent(
            memory_manager=self.memory_manager, **settings
        )

        # Register the settings update callback
        self.settings_manager.on_update(self._update_settings)

    def _update_settings(self):
        """
        Updates the MemoryAgent's models based on the latest settings.
        """
        settings = self.settings_manager.get_settings()
        if self.memory_agent:
            self.memory_agent.update_completion_model(**settings)
            self.memory_agent.update_embedding_model(**settings)
            logging.debug("MemoryAgentThread settings updated.")

    def start(self):
        """
        Starts the MemoryAgent processing thread.
        """
        if self.thread and self.thread.is_alive():
            logging.warning("MemoryAgentThread is already running.")
            return

        self.shutdown_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="MemoryAgentThread")
        self.thread.start()
        logging.info("MemoryAgentThread started.")

    def stop(self):
        """
        Signals the thread to shut down and waits for it to finish.
        """
        if not self.thread:
            logging.warning("MemoryAgentThread is not running.")
            return

        logging.info("Stopping MemoryAgentThread.")
        self.shutdown_event.set()
        # Put sentinel to unblock queue.get
        self.inbound_queue.put(None)
        self.thread.join()
        self.thread = None
        logging.info("MemoryAgentThread stopped.")

    def save_memories_async(self, observations: List[str], observation_images: List[Dict[str, PIL.Image.Image]]):
        """
        Enqueues observations and their corresponding images to be saved by the MemoryAgent.

        Args:
            observations (List[str]): A list of textual observations.
            observation_images (List[Dict[str, PIL.Image.Image]]): A list of dictionaries containing images related to observations.
                Each dictionary should have a consistent key structure, e.g., {"image_id": PIL.Image.Image}.
        """
        if not isinstance(observations, list) or not all(isinstance(obs, str) for obs in observations):
            logging.error("Invalid type for observations. Expected List[str].")
            raise ValueError("observations must be a list of strings.")

        if not isinstance(observation_images, list) or not all(isinstance(img_dict, dict) for img_dict in observation_images):
            logging.error("Invalid type for observation_images. Expected List[Dict[str, PIL.Image.Image]].")
            raise ValueError("observation_images must be a list of dictionaries.")

        message = {
            'observations': observations,
            'observation_images': observation_images
        }

        try:
            self.inbound_queue.put(message, block=False)
            logging.debug(f"Enqueued memories: {message}")
        except queue.Full:
            logging.error("Inbound queue is full. Failed to enqueue memories.")
            # Depending on requirements, you might want to handle this differently,
            # such as retrying after a delay or dropping the message.

    def _run(self):
        """
        The main loop that processes incoming messages and manages memory.
        """
        logging.info("MemoryAgentThread running.")

        try:
            while not self.shutdown_event.is_set():
                try:
                    # Attempt to get a message from the inbound_queue with a timeout
                    message = self.inbound_queue.get(timeout=0.05)  # 50ms timeout
                    if message is None:
                        # Sentinel received, exit the loop
                        logging.info("MemoryAgentThread received shutdown signal.")
                        break

                    logging.debug(f"MemoryAgentThread received message: {message}")

                    if self.memory_agent:
                        observations = message.get('observations', [])
                        observation_images = message.get('observation_images', [])
                        self.memory_agent.save_memories(
                            observations=observations,
                            observation_images=observation_images
                        )
                except queue.Empty:
                    continue  # No message received, continue the loop
                except Exception as e:
                    logging.error(f"Error in MemoryAgentThread: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"Exception in MemoryAgentThread: {e}", exc_info=True)

        finally:
            logging.info("MemoryAgentThread shutting down.")
