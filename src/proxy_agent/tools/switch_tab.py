import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)


class SwitchTabTool(Tool):
    name = "switch_tab"
    description = "Switch to a different browser tab by its page ID."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context.",
        },
        "page_id": {
            "type": "integer",
            "description": "The ID of the tab to switch to.",
        },
    }

    def __init__(self, on_observation: Callable[[str], None]):
        """
        Initialize the SwitchTabTool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        super().__init__()

    def forward(self, browser_context: BrowserContext, page_id: int):
        """
        Initiates the tab-switching operation in a separate thread.

        :param browser_context: The browser context.
        :param page_id: The ID of the tab to switch to.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"page_id: {page_id}."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous tab switching
        thread = threading.Thread(
            target=self._switch_tab_thread,
            args=(browser_context, page_id),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _switch_tab_thread(self, browser_context: BrowserContext, page_id: int):
        """
        The target method for the thread which handles the asynchronous tab-switching operation.

        :param browser_context: The browser context.
        :param page_id: The ID of the tab to switch to.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous switch tab method
            loop.run_until_complete(
                self._async_switch_tab(browser_context, page_id)
            )
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"An unexpected error occurred while switching tabs: {str(e)}."
            )
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_switch_tab(self, browser_context: BrowserContext, page_id: int) -> None:
        """
        Asynchronous method to switch to the specified tab and handle post-switch actions.

        :param browser_context: The browser context.
        :param page_id: The ID of the tab to switch to.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started switching to tab "
            f"with page_id {page_id}."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            # Attempt to switch to the specified tab
            await browser_context.switch_to_tab(page_id)
            logger.debug(f"Switched to tab with page_id {page_id}.")

            # Retrieve the current page after switching
            page = await browser_context.get_current_page()
            logger.debug("Retrieved the current page after switching.")

            # Wait for the page to fully load
            await page.wait_for_load_state()
            load_time = datetime.utcnow().isoformat() + "Z"
            success_message = (
                f"[{load_time}] SUCCESS in '{self.name}': "
                f"Successfully switched to tab with page_id {page_id} and the page has fully loaded."
            )
            logger.info(success_message)
            self.on_observation(success_message)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"Failed to switch to tab with page_id {page_id}. Error: {str(e)}."
            )
            logger.error(error_msg)
            self.on_observation(error_msg)
