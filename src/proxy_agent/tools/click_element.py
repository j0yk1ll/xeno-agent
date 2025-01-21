import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class ClickElementTool(Tool):
    name = "click_element"
    description = "Click an element on the page by its index and optional XPath."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context."
        },
        "index": {
            "type": "integer",
            "description": "The index of the element to click."
        },
        "xpath": {
            "type": "string",
            "description": "The XPath of the element to click (optional).",
            "nullable": True,
        },
    }

    def __init__(self, on_observation: Callable[[str], None]):
        """
        Initialize the ClickElementTool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        super().__init__()

    def forward(self, browser_context: BrowserContext, index: int, xpath: Optional[str] = None):
        """
        Initiates the click element operation in a separate thread.

        :param browser_context: The browser context.
        :param index: The index of the element to click.
        :param xpath: The XPath of the element to click (optional).
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"index: {index}, xpath: {xpath if xpath else 'not provided'}."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous click operation
        thread = threading.Thread(
            target=self._click_element_thread,
            args=(browser_context, index, xpath),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _click_element_thread(self, browser_context: BrowserContext, index: int, xpath: Optional[str]):
        """
        The target method for the thread which handles the asynchronous click operation.

        :param browser_context: The browser context.
        :param index: The index of the element to click.
        :param xpath: The XPath of the element to click (optional).
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous forward method
            result = loop.run_until_complete(
                self._async_forward(browser_context, index, xpath)
            )
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] An unexpected error occurred in '{self.name}': {str(e)}."
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_forward(self, browser_context: BrowserContext, index: int, xpath: Optional[str]) -> str:
        """
        The asynchronous method that performs the actual click operation.

        :param browser_context: The browser context.
        :param index: The index of the element to click.
        :param xpath: The XPath of the element to click (optional).
        :return: A message indicating the result of the click operation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started clicking the element "
            f"with index {index} and xpath '{xpath if xpath else 'not provided'}'."
        )
        logger.info(start_message)
        self.on_observation(start_message)
        
        try:
            session = await browser_context.get_session()
            state = session.cached_state

            if index not in state.selector_map:
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                    f"No element found with index {index}. Please verify the index or try alternative actions."
                )
                logger.error(msg)
                return msg

            element_node = state.selector_map[index]
            initial_pages = len(session.context.pages)

            if await browser_context.is_file_uploader(element_node):
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] INFO in '{self.name}': "
                    f"The element at index {index} is a file uploader. "
                    "Please use a dedicated file upload function to handle file uploads."
                )
                logger.info(msg)
                return msg

            try:
                await browser_context._click_element_node(element_node)
                clicked_element_text = element_node.get_all_text_till_next_clickable_element(max_depth=2)
                success_time = datetime.utcnow().isoformat() + "Z"
                msg = (
                    f"[{success_time}] SUCCESS in '{self.name}': "
                    f"Successfully clicked the element with index {index}: '{clicked_element_text}'."
                )
                logger.info(msg)
                logger.debug(f"Element XPath: {element_node.xpath}")

                # Check if a new tab was opened
                if len(session.context.pages) > initial_pages:
                    await browser_context.switch_to_tab(-1)
                    new_tab_time = datetime.utcnow().isoformat() + "Z"
                    new_tab_msg = (
                        f"[{new_tab_time}] INFO in '{self.name}': A new tab was opened and switched to successfully."
                    )
                    msg += f" {new_tab_msg}"
                    logger.info(new_tab_msg)
                return msg
            except Exception as e:
                error_time = datetime.utcnow().isoformat() + "Z"
                error_msg = (
                    f"[{error_time}] WARNING in '{self.name}': "
                    f"Failed to click the element with index {index}. Error: {str(e)}."
                )
                logger.warning(error_msg)
                return error_msg
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"An unexpected error occurred during the click operation: {str(e)}."
            )
            logger.error(error_msg)
            return error_msg
