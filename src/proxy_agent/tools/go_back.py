import logging
import threading
import asyncio
from typing import Callable
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class GoBackTool(Tool):
    name = "go_back"
    description = "Navigate back in the browser history of the current tab."
    inputs = {
        "browser_context": {"type": "object", "description": "The browser context."}
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initialize the GoBackTool with a callback function.

        :param on_observation: A callable that takes a single string argument to handle the result.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext):
        """
        Initiate the go back action in a separate thread.

        :param browser_context: The browser context to operate on.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"browser_context: {browser_context}."
        )
        logger.info(initiation_message)
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous navigation
        thread = threading.Thread(target=self._go_back_thread, args=(browser_context,), daemon=True)
        thread.start()

    def _go_back_thread(self, browser_context: BrowserContext):
        """
        The target method for the thread which handles the asynchronous navigation back.

        :param browser_context: The browser context to operate on.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the coroutine to navigate back
            result = loop.run_until_complete(self._navigate_back(browser_context))
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': Failed to navigate back. Error: {str(e)}."
            )
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _navigate_back(self, browser_context: BrowserContext) -> str:
        """
        Asynchronously navigate back in the browser history.

        :param browser_context: The browser context to operate on.
        :return: A message indicating the result of the navigation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started navigating back in the browser history."
        )
        logger.info(start_message)
        self.on_observation(start_message)
        
        try:
            page = await browser_context.get_current_page()
            initial_url = page.url
            logger.debug(f"Initial URL before going back: {initial_url}")

            await page.go_back()
            await page.wait_for_load_state()

            new_url = page.url
            success_time = datetime.utcnow().isoformat() + "Z"
            msg = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f"Successfully navigated back from '{initial_url}' to '{new_url}'."
            )
            logger.info(msg)
            return msg
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"An error occurred while navigating back: {str(e)}."
            )
            logger.error(error_msg)
            raise
