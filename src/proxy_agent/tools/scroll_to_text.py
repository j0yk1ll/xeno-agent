import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)


class ScrollToTextTool(Tool):
    name = "scroll_to_text"
    description = "Scroll the page to bring the specified text into view."
    inputs = {
        "browser_context": {"type": "object", "description": "The browser context."},
        "text": {"type": "string", "description": "The text to scroll to."},
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initializes the ScrollToTextTool with a callback function.

        Args:
            on_observation (Callable[[str], None]): A callback function to handle the result.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, text: str):
        """
        Initiates the scrolling operation in a separate thread.

        Args:
            browser_context (BrowserContext): The browser context.
            text (str): The text to scroll to.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"text: '{text}'."
        )
        logger.info(initiation_message)
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous scrolling
        thread = threading.Thread(
            target=self._scroll_to_text_thread,
            args=(browser_context, text),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _scroll_to_text_thread(self, browser_context: BrowserContext, text: str):
        """
        The target method for the thread which handles the asynchronous scrolling operation.

        Args:
            browser_context (BrowserContext): The browser context.
            text (str): The text to scroll to.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous scroll method
            result = loop.run_until_complete(
                self._async_scroll(browser_context, text)
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

    async def _async_scroll(self, browser_context: BrowserContext, text: str) -> str:
        """
        Asynchronously scrolls to the specified text on the page.

        Args:
            browser_context (BrowserContext): The browser context.
            text (str): The text to scroll to.

        Returns:
            str: A message indicating the result of the scrolling operation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started scrolling to the text: '{text}'."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            page = await browser_context.get_current_page()
            locators = [
                page.get_by_text(text, exact=False),
                page.locator(f"text={text}"),
                page.locator(f"//*[contains(text(), '{text}')]" ),
            ]

            for idx, locator in enumerate(locators, start=1):
                try:
                    if await locator.count() > 0 and await locator.first.is_visible():
                        await locator.first.scroll_into_view_if_needed()
                        await asyncio.sleep(0.5)  # Wait for scroll to complete
                        success_time = datetime.utcnow().isoformat() + "Z"
                        msg = (
                            f"[{success_time}] SUCCESS in '{self.name}': "
                            f"Scrolled to text: '{text}' using locator method {idx}."
                        )
                        logger.info(msg)
                        return msg
                except Exception as e:
                    debug_time = datetime.utcnow().isoformat() + "Z"
                    logger.debug(
                        f"[{debug_time}] DEBUG in '{self.name}': Locator method {idx} failed with error: {str(e)}."
                    )
                    continue

            not_found_time = datetime.utcnow().isoformat() + "Z"
            msg = (
                f"[{not_found_time}] WARNING in '{self.name}': "
                f"Text '{text}' not found or not visible on the page."
            )
            logger.warning(msg)
            return msg
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': {str(e)}."
            logger.error(error_msg)
            return error_msg
