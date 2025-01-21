import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class SearchGoogleTool(Tool):
    name = "search_google"
    description = "Search for a query on Google in the current browser tab."
    inputs = {
        "browser_context": {"type": "object", "description": "The browser context."},
        "query": {
            "type": "string",
            "description": "The search query to perform on Google.",
        },
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initialize the SearchGoogleTool with a callback function.

        Args:
            on_observation (Callable[[str], None]): A callback function to handle the result.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, query: str):
        """
        Initiates the Google search operation in a separate thread.

        Args:
            browser_context (BrowserContext): The browser context to use.
            query (str): The search query.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"query: '{query}'."
        )
        logger.info(initiation_message)
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous search operation
        thread = threading.Thread(
            target=self._perform_search_thread,
            args=(browser_context, query),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _perform_search_thread(self, browser_context: BrowserContext, query: str):
        """
        The target method for the thread which handles the asynchronous search operation.

        Args:
            browser_context (BrowserContext): The browser context to use.
            query (str): The search query.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous search coroutine
            result = loop.run_until_complete(self._async_search(browser_context, query))
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

    async def _async_search(self, browser_context: BrowserContext, query: str) -> str:
        """
        Asynchronously performs the Google search.

        Args:
            browser_context (BrowserContext): The browser context to use.
            query (str): The search query.

        Returns:
            str: A message indicating the result of the search.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started searching for "
            f"'{query}' on Google."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            page = await browser_context.get_current_page()
            await page.goto(f"https://www.google.com/search?q={query}&udm=14")
            await page.wait_for_load_state()

            success_time = datetime.utcnow().isoformat() + "Z"
            success_message = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f"Searched for '{query}' on Google successfully."
            )
            logger.info(success_message)
            self.on_observation(success_message)

            # Optionally, you can capture more details like the number of results
            # or any other relevant information here.

            return success_message
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': Failed to search for '{query}'. Error: {str(e)}."
            logger.error(error_msg)
            return error_msg
