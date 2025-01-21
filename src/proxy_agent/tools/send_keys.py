import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class SendKeysTool(Tool):
    name = "send_keys"
    description = "Send a sequence of keyboard keys to the current page."
    inputs = {
        "browser_context": {"type": "object", "description": "The browser context."},
        "keys": {
            "type": "string",
            "description": "The keys to send (e.g., 'Enter', 'Control+o').",
        },
    }

    def __init__(self, on_observation: Callable[[str], None]):
        """
        Initialize the SendKeysTool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        super().__init__()

    def forward(self, browser_context: BrowserContext, keys: str):
        """
        Initiates the send keys operation in a separate thread.

        :param browser_context: The browser context.
        :param keys: The keys to send.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"keys: '{keys}'."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous key sending
        thread = threading.Thread(
            target=self._send_keys_thread,
            args=(browser_context, keys),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _send_keys_thread(self, browser_context: BrowserContext, keys: str):
        """
        The target method for the thread which handles the asynchronous send keys operation.

        :param browser_context: The browser context.
        :param keys: The keys to send.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous send keys method
            result = loop.run_until_complete(
                self._async_send_keys(browser_context, keys)
            )
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': {str(e)}."
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_send_keys(self, browser_context: BrowserContext, keys: str) -> str:
        """
        The asynchronous method that performs the actual send keys operation.

        :param browser_context: The browser context.
        :param keys: The keys to send.
        :return: A message indicating the result of the send keys operation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started sending keys "
            f"'{keys}' to the current page."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            page = await browser_context.get_current_page()
            if not page:
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                    "No active page found in the browser context."
                )
                logger.error(msg)
                return msg

            await page.keyboard.press(keys)
            success_time = datetime.utcnow().isoformat() + "Z"
            msg = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f"Successfully sent keys '{keys}' to the current page."
            )
            logger.info(msg)
            return msg
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] WARNING in '{self.name}': "
                f"Failed to send keys '{keys}'. Error: {str(e)}."
            )
            logger.warning(error_msg)
            return error_msg
