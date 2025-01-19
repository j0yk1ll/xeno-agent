import logging
from typing import Dict, List, Optional

import PIL

from src.utils.memory_manager import MemoryManager
from src.utils.settings_manager import SettingsManager
from src.utils.threads.memory_agent_thread import MemoryAgentThread


class MemoryAgentThreadManager:
    """
    Class that manages the single instance of MemoryAgentThread.
    Ensures only one MemoryAgentThread is active at any time.
    """
    
    def __init__(
        self,
        settings_manager: SettingsManager,
        memory_manager: MemoryManager,
    ):
        """
        Initializes the MemoryAgentThreadManager with the necessary managers.
        
        Args:
            settings_manager (SettingsManager): Manager for application settings.
            memory_manager (MemoryManager): Manager for memory operations.
        """
        self.settings_manager = settings_manager
        self.memory_manager = memory_manager
        self.memory_agent_thread: Optional[MemoryAgentThread] = None
        self._initialize_memory_agent_thread()
    
    def _initialize_memory_agent_thread(self):
        """
        Initializes the MemoryAgentThread instance.
        """
        if not self.memory_agent_thread:
            self.memory_agent_thread = MemoryAgentThread(
                settings_manager=self.settings_manager,
                memory_manager=self.memory_manager,
            )
            logging.info("MemoryAgentThread initialized.")
    
    def start(self):
        """
        Starts the MemoryAgentThread if it's not already running.
        """
        if self.memory_agent_thread:
            self.memory_agent_thread.start()
            logging.info("MemoryAgentThread started.")
        else:
            logging.error("MemoryAgentThread is not initialized.")
    
    def stop(self):
        """
        Stops the MemoryAgentThread if it's running.
        """
        if self.memory_agent_thread:
            self.memory_agent_thread.stop()
            self.memory_agent_thread = None
            logging.info("MemoryAgentThread stopped.")
        else:
            logging.warning("MemoryAgentThread is not running.")
    
    def save_memories_async(
        self,
        observations: List[str],
        observation_images: List[Dict[str, PIL.Image]],
    ):
        """
        Enqueues memories to be saved by the MemoryAgentThread.
        
        Args:
            observations (List[str]): A list of textual observations.
            observation_images (List[Dict[str, PIL.Image.Image]]): A list of dictionaries containing images related to observations.
        """
        if self.memory_agent_thread:
            self.memory_agent_thread.save_memories_async(
                observations=observations,
                observation_images=observation_images,
            )
            logging.debug("Memories enqueued successfully.")
        else:
            logging.error("MemoryAgentThread is not running. Cannot enqueue memories.")
            raise RuntimeError("MemoryAgentThread is not running.")
    
    def shutdown(self):
        """
        Shuts down the MemoryAgentThread gracefully.
        """
        self.stop()
        logging.info("MemoryAgentThreadManager has been shutdown.")
