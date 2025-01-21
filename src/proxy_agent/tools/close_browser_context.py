import logging
import threading
import asyncio
from typing import Callable
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

# Configure the logger for this module
logger = logging.getLogger(__name__)

class CloseBrowserContextTool(Tool):
    name = "close_browser_context"
    description = "Close an open browser context."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context."
        }
    }

    def __init__(self, on_observation: Callable[[str], None]):
        """
        Initialize the CloseBrowserContextTool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        super().__init__()

    def forward(self, browser_context: BrowserContext):
        """
        Initiates the close browser context operation in a separate thread.

        :param browser_context: The browser context to close.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool to close the browser context."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous closure
        thread = threading.Thread(
            target=self._close_context_thread,
            args=(browser_context,),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _close_context_thread(self, browser_context: BrowserContext):
        """
        The target method for the thread which handles the asynchronous closure of the browser context.

        :param browser_context: The browser context to close.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous close operation
            result = loop.run_until_complete(self._async_close_context(browser_context))
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"An unexpected error occurred while closing the browser context: {str(e)}."
            )
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_close_context(self, browser_context: BrowserContext) -> str:
        """
        The asynchronous method that performs the actual closure of the browser context.

        :param browser_context: The browser context to close.
        :return: A message indicating the result of the closure operation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started closing the browser context."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            # Attempt to close the browser context asynchronously
            await browser_context.close()
            success_time = datetime.utcnow().isoformat() + "Z"
            success_msg = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                "The browser context was closed successfully."
            )
            logger.info(success_msg)
            return success_msg
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"Failed to close the browser context. Error: {str(e)}."
            )
            logger.error(error_msg)
            return error_msg
