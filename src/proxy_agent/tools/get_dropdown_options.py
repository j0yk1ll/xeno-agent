import logging
import threading
import asyncio
import json
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class GetDropdownOptionsTool(Tool):
    name = "get_dropdown_options"
    description = "Retrieve all options from a native dropdown element by its index."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context."
        },
        "index": {
            "type": "integer",
            "description": "The index of the dropdown element."
        },
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initialize the tool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, index: int):
        """
        Initiate the retrieval of dropdown options in a separate thread.

        :param browser_context: The browser context.
        :param index: The index of the dropdown element.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"index: {index}."
        )
        logger.info(initiation_message)
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous operation
        thread = threading.Thread(target=self._retrieve_options, args=(browser_context, index), daemon=True)
        thread.start()

    def _retrieve_options(self, browser_context: BrowserContext, index: int):
        """
        This method runs in a separate thread and handles the asynchronous
        retrieval of dropdown options.

        :param browser_context: The browser context.
        :param index: The index of the dropdown element.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            start_time = datetime.utcnow().isoformat() + "Z"
            start_message = f"[{start_time}] Starting asynchronous retrieval of dropdown options."
            logger.info(start_message)
            self.on_observation(start_message)

            # Run the asynchronous coroutine
            result = loop.run_until_complete(self._get_dropdown_options(browser_context, index))

            completion_time = datetime.utcnow().isoformat() + "Z"
            completion_message = f"[{completion_time}] Completed retrieval of dropdown options."
            logger.info(completion_message)
            self.on_observation(completion_message)

            # Call the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': {str(e)}"
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _get_dropdown_options(self, browser_context: BrowserContext, index: int) -> str:
        """
        Asynchronously retrieve dropdown options.

        :param browser_context: The browser context.
        :param index: The index of the dropdown element.
        :return: A message containing all dropdown options or an error message.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started processing dropdown at index {index}."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            page = await browser_context.get_current_page()
            selector_map = await browser_context.get_selector_map()
            dom_element = selector_map.get(index)

            if not dom_element:
                msg_time = datetime.utcnow().isoformat() + "Z"
                msg = f"[{msg_time}] ERROR in '{self.name}': No dropdown found with index {index}."
                logger.error(msg)
                return msg

            all_options = []
            frame_index = 0

            for frame in page.frames:
                frame_time = datetime.utcnow().isoformat() + "Z"
                logger.debug(f"[{frame_time}] Evaluating frame {frame_index} for dropdown options.")

                try:
                    options = await frame.evaluate(
                        """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;

                            return {
                                options: Array.from(select.options).map(opt => ({
                                    text: opt.text,
                                    value: opt.value,
                                    index: opt.index
                                })),
                                id: select.id,
                                name: select.name
                            };
                        }
                        """,
                        dom_element.xpath,
                    )

                    if options:
                        option_found_time = datetime.utcnow().isoformat() + "Z"
                        logger.debug(f"[{option_found_time}] Found dropdown in frame {frame_index}")
                        logger.debug(f"[{option_found_time}] Dropdown ID: {options['id']}, Name: {options['name']}")

                        formatted_options = []
                        for opt in options['options']:
                            encoded_text = json.dumps(opt['text'])
                            formatted_options.append(f"{opt['index']}: text={encoded_text}")

                        all_options.extend(formatted_options)
                        # Optionally, break if dropdown is found
                        break

                except Exception as frame_e:
                    frame_error_time = datetime.utcnow().isoformat() + "Z"
                    logger.debug(f"[{frame_error_time}] Frame {frame_index} evaluation failed: {str(frame_e)}")

                frame_index += 1

            if all_options:
                options_time = datetime.utcnow().isoformat() + "Z"
                msg = (
                    f"[{options_time}] SUCCESS in '{self.name}': Retrieved dropdown options:\n" +
                    '\n'.join(all_options) +
                    "\nUse the exact text string in 'select_dropdown_option'."
                )
                logger.info(msg)
                return msg
            else:
                no_options_time = datetime.utcnow().isoformat() + "Z"
                msg = f"[{no_options_time}] INFO in '{self.name}': No options found in any frame for dropdown at index {index}."
                logger.info(msg)
                return msg

        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': An unexpected error occurred: {str(e)}."
            logger.error(error_msg)
            return error_msg
