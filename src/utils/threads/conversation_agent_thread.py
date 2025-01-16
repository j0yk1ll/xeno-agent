import logging
import queue
import threading

from src.conversation_agent.agent import ConversationAgent
from src.utils.settings_manager import SettingsManager
from src.utils.threads.task_agent_tread_manager import TaskAgentThreadManager


def conversation_agent_thread(
    inbound_queue: queue.Queue,
    outbound_queue: queue.Queue,
    shutdown_event: threading.Event,
    settings_manager: SettingsManager,
):
    """
    Entry point for the conversation agent thread.
    Runs the ConversationAgent in a background thread, communicating via queue.Queue.
    """

    def _update_settings():
        settings = settings_manager.get_settings()
        # Update TaskAgentManager
        task_agent_thread_manager.update_settings(settings)

        conversation_agent.update_completion_model(**settings)
        conversation_agent.update_embedding_model(**settings)

    settings = settings_manager.get_settings()

    settings_manager.on_update(_update_settings)

    # Initialize the TaskAgentThreadManager
    task_agent_thread_manager = TaskAgentThreadManager(settings)

    # Initialize the ConversationAgent
    conversation_agent = ConversationAgent(
        task_agent_thread_manager=task_agent_thread_manager, **settings
    )

    def put_response(response: str):
        logging.debug(f"Received response from conversation agent: {response}")
        outbound_queue.put(response)

    conversation_agent.callback = put_response

    try:
        while not shutdown_event.is_set():
            try:
                # Attempt to get a message from the inbound_queue with a timeout
                message = inbound_queue.get(timeout=0.05)  # 50ms timeout
                if message is None:
                    # Sentinel received, exit the loop
                    break

                # Add observation
                conversation_agent.add_observation(message)
            except queue.Empty:
                # No message received, continue the loop
                continue
    except Exception as e:
        # Log or handle exceptions as needed
        logging.warning(f"Conversation Agent encountered an error: {e}")
    finally:
        # Ensure that resources are cleaned up properly
        task_agent_thread_manager.shutdown()
        logging.warning("Conversation Agent Thread shutting down.")
