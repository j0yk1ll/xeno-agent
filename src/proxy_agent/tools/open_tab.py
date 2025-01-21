import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class OpenTabTool(Tool):
    name = "open_tab"
    description = "Open a new browser tab with the specified URL."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context."
        },
        "url": {"type": "string", "description": "The URL to open in the new tab."},
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initializes the OpenTabTool with a callback function.

        :param on_observation: A callable that takes a string message as its argument.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, url: str):
        """
        Initiates the process to open a new browser tab by starting a separate thread.

        :param browser_context: The browser context in which to open the tab.
        :param url: The URL to open in the new tab.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"url: '{url}'."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous tab opening
        thread = threading.Thread(
            target=self._open_tab_thread,
            args=(browser_context, url),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _open_tab_thread(self, browser_context: BrowserContext, url: str):
        """
        The target method for the thread which handles the asynchronous open tab operation.

        :param browser_context: The browser context to use.
        :param url: The URL to open.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous forward method
            result = loop.run_until_complete(
                self._async_forward(browser_context, url)
            )
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"An unexpected error occurred while opening the tab: {str(e)}."
            )
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_forward(self, browser_context: BrowserContext, url: str) -> str:
        """
        The asynchronous method that performs the actual open tab operation.

        :param browser_context: The browser context.
        :param url: The URL to open.
        :return: A message indicating the result of the open tab operation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started opening a new tab with URL '{url}'."
        )
        logger.info(start_message)
        self.on_observation(start_message)
        
        try:
            # Attempt to open the new tab
            await browser_context.create_new_tab(url)
            success_time = datetime.utcnow().isoformat() + "Z"
            success_message = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f"Successfully opened a new tab with URL '{url}'."
            )
            logger.info(success_message)
            return success_message
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_message = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"Failed to open a new tab with URL '{url}'. Error: {str(e)}."
            )
            logger.error(error_message)
            return error_message
