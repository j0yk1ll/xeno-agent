import logging
import threading
import asyncio
from typing import Optional, Callable
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class ScrollPageTool(Tool):
    name = "scroll_page"
    description = "Scroll the page up or down by a specified pixel amount or by one page."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context."
        },
        "direction": {
            "type": "string",
            "description": "Direction to scroll: 'up' or 'down'."
        },
        "amount": {
            "type": "integer",
            "description": "Number of pixels to scroll. If not specified, scroll one page. (optional)",
            "nullable": True,
        },
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initialize the ScrollPageTool with a callback function.

        :param on_observation: A callable that takes a string argument to handle the result.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, direction: str, amount: Optional[int]):
        """
        Initiate the scroll operation in a separate thread.

        :param browser_context: The browser context.
        :param direction: Direction to scroll: 'up' or 'down'.
        :param amount: Number of pixels to scroll. If None, scroll one page.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"direction: '{direction}', amount: {amount if amount is not None else 'one page'}, "
            f"browser_context: {browser_context}."
        )
        logger.info(initiation_message)
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous scrolling operation
        thread = threading.Thread(
            target=self._scroll_page_thread,
            args=(browser_context, direction, amount),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _scroll_page_thread(self, browser_context: BrowserContext, direction: str, amount: Optional[int]):
        """
        The target method for the thread which handles the asynchronous scroll operation.

        :param browser_context: The browser context.
        :param direction: Direction to scroll: 'up' or 'down'.
        :param amount: Number of pixels to scroll. If None, scroll one page.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            operation_start_time = datetime.utcnow().isoformat() + "Z"
            start_message = (
                f"[{operation_start_time}] The '{self.name}' tool has started scrolling "
                f"the page {direction} by "
                f"{amount if amount is not None else 'one page'}."
            )
            logger.info(start_message)
            self.on_observation(start_message)

            # Run the asynchronous scroll operation
            result = loop.run_until_complete(
                self._perform_scroll(browser_context, direction, amount)
            )
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': An unexpected error occurred during the scroll operation: {str(e)}."
            )
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _perform_scroll(self, browser_context: BrowserContext, direction: str, amount: Optional[int]) -> str:
        """
        Asynchronously perform the scroll operation.

        :param browser_context: The browser context.
        :param direction: Direction to scroll: 'up' or 'down'.
        :param amount: Number of pixels to scroll. If None, scroll one page.
        :return: Result message of the scroll operation.
        """
        try:
            page = await browser_context.get_current_page()
            if direction.lower() == "down":
                if amount is not None:
                    await page.evaluate(f'window.scrollBy(0, {amount});')
                    msg = f"[{datetime.utcnow().isoformat()}Z] SUCCESS in '{self.name}': Scrolled down the page by {amount} pixels."
                else:
                    await page.keyboard.press('PageDown')
                    msg = f"[{datetime.utcnow().isoformat()}Z] SUCCESS in '{self.name}': Scrolled down the page by one page."
            elif direction.lower() == "up":
                if amount is not None:
                    await page.evaluate(f'window.scrollBy(0, -{amount});')
                    msg = f"[{datetime.utcnow().isoformat()}Z] SUCCESS in '{self.name}': Scrolled up the page by {amount} pixels."
                else:
                    await page.keyboard.press('PageUp')
                    msg = f"[{datetime.utcnow().isoformat()}Z] SUCCESS in '{self.name}': Scrolled up the page by one page."
            else:
                msg = f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': Invalid direction '{direction}'. Use 'up' or 'down'."
                logger.error(msg)
                return msg

            logger.info(msg)
            return msg
        except Exception as e:
            error_msg = f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': Failed to scroll the page. Error: {str(e)}."
            logger.error(error_msg)
            return error_msg
