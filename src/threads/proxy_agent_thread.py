import logging
import queue
import threading
from typing import Optional

from src.proxy_agent.agent import ProxyAgent
from src.utils.settings_manager import SettingsManager
from src.utils.browser.browser import Browser
from src.utils.llms.completion import CompletionLLM


class ProxyAgentThread:
    """
    Class that manages the ProxyAgent in a separate thread.
    """

    def __init__(
        self,
        inbound_queue: queue.Queue,
        outbound_queue: queue.Queue,
        settings_manager: SettingsManager,
    ):
        """
        Initializes the ProxyAgentThread.

        Args:
            inbound_queue (queue.Queue): Queue from which to receive messages.
            outbound_queue (queue.Queue): Queue to send responses.
            settings_manager (SettingsManager): Manager for application settings.
        """
        self.inbound_queue = inbound_queue
        self.outbound_queue = outbound_queue
        self.settings_manager = settings_manager
        self.shutdown_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        # Load initial settings
        settings = self.settings_manager.get_settings()

        self.browser = Browser()

        completion_llm = CompletionLLM(
            model_id=settings['completion_model_id'],
            api_base=settings['completion_api_base'],
            api_key=settings['completion_api_key'],
            requests_per_minute=5,
        )

        # Initialize the ProxyAgent
        self.proxy_agent = ProxyAgent(
            browser=self.browser,
            completion_llm=completion_llm,
            on_response=self._put_response,
        )

        # Register the settings update callback
        self.settings_manager.on_update(self._update_settings)

        logging.debug("ProxyAgentThread instance initialized.")

    def _put_response(self, response: str):
        """
        Callback function to handle responses from the ProxyAgent.
        """
        logging.debug(f"ProxyAgentThread received response: {response}")
        try:
            self.outbound_queue.put(response, timeout=0.1)
            logging.debug("Response put into outbound queue.")
        except queue.Full:
            logging.warning("Outbound queue is full. Dropping response.")

    def _update_settings(self):
        """
        Update settings for TaskAgentThreadManager and ProxyAgent.
        """
        settings = self.settings_manager.get_settings()
        logging.debug("ProxyAgentThread updating settings.")

        completion_llm = CompletionLLM(
            model_id=settings['completion_model_id'],
            api_base=settings['completion_api_base'],
            api_key=settings['completion_api_key'],
            requests_per_minute=5,
        )

        # Update ProxyAgent models
        self.proxy_agent.update_completion_llm(completion_llm)
        logging.debug("ProxyAgent models updated with new settings.")

    def start(self):
        """
        Starts the ProxyAgent processing thread.
        """
        if self.thread and self.thread.is_alive():
            logging.warning("ProxyAgentThread is already running.")
            return

        self.shutdown_event.clear()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="ProxyAgentThread"
        )
        self.thread.start()
        logging.debug(
            f"Thread '{self.thread.name}' started with ID {self.thread.ident}."
        )
        logging.info("ProxyAgentThread started.")

    def stop(self):
        """
        Signals the ProxyAgent thread to shut down.
        """
        if not self.thread:
            logging.warning("ProxyAgentThread is not running.")
            return

        logging.info("Stopping ProxyAgentThread.")
        self.shutdown_event.set()
        self.inbound_queue.put(None)  # Send sentinel to unblock the queue

        if self.thread.is_alive():
            self.thread.join()
            logging.debug(
                f"Thread '{self.thread.name}' with ID {self.thread.ident} has been joined and stopped."
            )
        else:
            logging.debug(
                f"Thread '{self.thread.name}' is not alive and does not need to be joined."
            )

        self.thread = None
        logging.info("ProxyAgentThread stopped.")

    def _run(self):
        """
        Main loop for the ProxyAgent thread.
        """
        logging.info("ProxyAgentThread running.")

        try:
            while not self.shutdown_event.is_set():
                try:
                    # Attempt to get a message from the inbound_queue with a timeout
                    message = self.inbound_queue.get(timeout=0.05)  # 50ms timeout
                    if message is None:
                        # Sentinel received, exit the loop
                        logging.info("ProxyAgentThread received shutdown signal.")
                        break

                    logging.debug(f"ProxyAgentThread received message: {message}")

                    # Add observation to ProxyAgent
                    message_text = message["text"]
                    message_source = message.get("source", "unknown")
                    message_files = message.get("files", [])
                    self.proxy_agent.add_observation(
                        message_source, message_text, message_files
                    )
                    self.inbound_queue.task_done()
                except queue.Empty:
                    continue  # No message received, continue the loop
                except Exception as e:
                    logging.error(f"Error in ProxyAgentThread: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"Exception in ProxyAgentThread: {e}", exc_info=True)

        finally:
            # Ensure that resources are cleaned up properly
            self.browser.close()
            logging.info("ProxyAgentThread shutting down.")
