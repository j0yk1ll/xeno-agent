import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)


class InputTextTool(Tool):
    name = "input_text"
    description = (
        "Input text into an interactive element by its index and optional XPath."
    )
    inputs = {
        "browser_context": {"type": "object", "description": "The browser context."},
        "index": {"type": "integer", "description": "The index of the input element."},
        "text": {
            "type": "string",
            "description": "The text to input into the element.",
        },
        "xpath": {
            "type": "string",
            "description": "The XPath of the input element (optional).",
            "nullable": True,
        },
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initialize the InputTextTool with a callback function.

        :param on_observation: A callable that takes a single string argument to handle the result.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(
        self,
        browser_context: BrowserContext,
        index: int,
        text: str,
        xpath: Optional[str] = None,
    ):
        """
        Initiate the input text action in a separate thread.

        :param browser_context: The browser context to operate on.
        :param index: The index of the input element.
        :param text: The text to input into the element.
        :param xpath: The XPath of the input element (optional).
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"index: {index}, text: '{text}', xpath: '{xpath if xpath else 'not provided'}'."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous input action
        thread = threading.Thread(
            target=self._input_text_thread, args=(browser_context, index, text, xpath), daemon=True
        )
        thread.start()

    def _input_text_thread(
        self,
        browser_context: BrowserContext,
        index: int,
        text: str,
        xpath: Optional[str],
    ):
        """
        The target method for the thread which handles the asynchronous input text operation.

        :param browser_context: The browser context to operate on.
        :param index: The index of the input element.
        :param text: The text to input into the element.
        :param xpath: The XPath of the input element (optional).
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous input text method
            result = loop.run_until_complete(
                self._async_input_text(browser_context, index, text, xpath)
            )
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': Failed to input text. Error: {e}."
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_input_text(
        self,
        browser_context: BrowserContext,
        index: int,
        text: str,
        xpath: Optional[str],
    ) -> str:
        """
        Asynchronously input text into the specified browser element.

        :param browser_context: The browser context to operate on.
        :param index: The index of the input element.
        :param text: The text to input into the element.
        :param xpath: The XPath of the input element (optional).
        :return: A message indicating the result of the input operation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started inputting text "
            f"into the element with index {index} and xpath '{xpath if xpath else 'not provided'}'."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            session = await browser_context.get_session()
            state = session.cached_state

            if index not in state.selector_map:
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                    f"No input element found with index {index}. Please verify the index or try alternative actions."
                )
                logger.error(msg)
                return msg

            element_node = state.selector_map[index]

            # If XPath is provided, verify it matches the element's XPath
            if xpath and element_node.xpath != xpath:
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                    f"Provided XPath does not match the element at index {index}. Expected XPath: '{element_node.xpath}', "
                    f"Provided XPath: '{xpath}'."
                )
                logger.error(msg)
                return msg

            await browser_context.input_text_element_node(element_node, text)
            success_time = datetime.utcnow().isoformat() + "Z"
            msg = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f'Successfully input "{text}" into element with index {index}.'
            )
            logger.info(msg)
            logger.debug(f"Element XPath: {element_node.xpath}")
            return msg

        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"An unexpected error occurred during the input operation: {e}."
            )
            logger.error(error_msg)
            return error_msg
