import logging
import threading
from typing import Callable, Dict
import weakref

from src.utils.threads.task_agent_thread import TaskAgentThread


class TaskAgentThreadManager:
    """
    Class that manages multiple TaskAgents. Each task is handled by a separate TaskAgentThread.
    Automatically releases agent_threads after task completion.
    """

    def __init__(
        self,
        settings: Dict
    ):
        """
        Initialize the TaskAgentThreadManager with initial settings.
        """
        # Store the current settings
        self.settings = settings

        # Keep track of active TaskAgentThreads using a WeakSet to allow garbage collection
        self.agent_threads = weakref.WeakSet()
        self.agent_threads_lock = threading.Lock()

    def run_task_async(self, task: str, callback: Callable[[str], None]) -> None:
        """
        Asynchronously run a task by spawning a new TaskAgentThread.

        :param task: The task to send to the agent.
        :param callback: A callable that takes the result string as its only argument.
        """
        def on_complete(agent_thread: TaskAgentThread):
            with self.agent_threads_lock:
                self.agent_threads.discard(agent_thread)

        # Create a new TaskAgentThread with the current configuration
        try:
            agent_thread = TaskAgentThread(
                task=task,
                callback=callback,
                on_complete=on_complete,
                **self.settings
            )
            with self.agent_threads_lock:
                self.agent_threads.add(agent_thread)
        except Exception as e:
            logging.error(f"Failed to run task asynchronously: {e}")
            callback(f"Error: {str(e)}")

    def shutdown(self):
        """
        Gracefully shutdown all active TaskAgentThreads.
        """
        with self.agent_threads_lock:
            agent_threads_copy = list(self.agent_threads)

        for agent_thread in agent_threads_copy:
            try:
                agent_thread.shutdown()
            except Exception as e:
                logging.error(f"Error shutting down TaskAgentThread: {e}")

        with self.agent_threads_lock:
            self.agent_threads.clear()

    def update_settings(
        self,
        settings: Dict
    ):
        self.settings = settings

        logging.info("TaskAgentThreadManager settings updated for future agent_threads.")
