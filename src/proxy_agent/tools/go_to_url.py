import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class GoToUrlTool(Tool):
    name = "go_to_url"
    description = "Navigate the current browser tab to a specified URL."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context."
        },
        "url": {
            "type": "string",
            "description": "The URL to navigate to."
        },
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initializes the GoToUrlTool with a callback function.

        Args:
            on_observation (Callable[[str], None]): A callback function to handle the result.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, url: str):
        """
        Initiates the navigation to the specified URL in a separate thread.

        Args:
            browser_context (BrowserContext): The browser context.
            url (str): The URL to navigate to.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"url: {url}."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous navigation
        thread = threading.Thread(
            target=self._navigate_to_url_thread,
            args=(browser_context, url),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _navigate_to_url_thread(self, browser_context: BrowserContext, url: str):
        """
        The target method for the thread which handles the asynchronous navigation operation.

        Args:
            browser_context (BrowserContext): The browser context.
            url (str): The URL to navigate to.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous navigate method
            result = loop.run_until_complete(
                self._async_navigate(browser_context, url)
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

    async def _async_navigate(self, browser_context: BrowserContext, url: str) -> str:
        """
        Asynchronous coroutine to navigate to the specified URL.

        Args:
            browser_context (BrowserContext): The browser context.
            url (str): The URL to navigate to.

        Returns:
            str: A message indicating the result of the navigation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started navigating to URL: {url}."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            session = await browser_context.get_session()
            initial_pages = len(session.context.pages)

            page = await browser_context.get_current_page()
            await page.goto(url)
            await page.wait_for_load_state()

            success_time = datetime.utcnow().isoformat() + "Z"
            success_msg = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f"Successfully navigated to {url}."
            )
            logger.info(success_msg)
            logger.debug(f"Page URL after navigation: {page.url}")

            # Check if a new tab was opened
            if len(session.context.pages) > initial_pages:
                await browser_context.switch_to_tab(-1)
                new_tab_time = datetime.utcnow().isoformat() + "Z"
                new_tab_msg = (
                    f"[{new_tab_time}] INFO in '{self.name}': A new tab was opened and switched to successfully."
                )
                success_msg += f" {new_tab_msg}"
                logger.info(new_tab_msg)

            return success_msg
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"Failed to navigate to {url}. Error: {str(e)}."
            )
            logger.error(error_msg)
            return error_msg
