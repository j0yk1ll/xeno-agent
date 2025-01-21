import logging
import threading
import asyncio
from typing import Callable, Optional
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext

logger = logging.getLogger(__name__)

class SelectDropdownOptionTool(Tool):
    name = "select_dropdown_option"
    description = (
        "Select an option in a dropdown element by its index and the option's text."
    )
    inputs = {
        "browser_context": {"type": "object", "description": "The browser context."},
        "index": {
            "type": "integer",
            "description": "The index of the dropdown element.",
        },
        "text": {"type": "string", "description": "The text of the option to select."},
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initializes the SelectDropdownOptionTool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, index: int, text: str):
        """
        Initiates the selection of a dropdown option in a separate thread.

        :param browser_context: The browser context.
        :param index: The index of the dropdown element.
        :param text: The text of the option to select.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"index: {index}, text: '{text}'."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous selection
        thread = threading.Thread(
            target=self._select_option_thread, 
            args=(browser_context, index, text),
            daemon=True  # Daemonize thread to ensure it exits with the main program
        )
        thread.start()

    def _select_option_thread(self, browser_context: BrowserContext, index: int, text: str):
        """
        The target method for the thread which handles the asynchronous selection.

        :param browser_context: The browser context.
        :param index: The index of the dropdown element.
        :param text: The text of the option to select.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous selection coroutine
            result = loop.run_until_complete(
                self._async_select_option(browser_context, index, text)
            )
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': Failed to select option '{text}'. Error: {str(e)}."
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_select_option(
        self, browser_context: BrowserContext, index: int, text: str
    ) -> str:
        """
        Asynchronously selects an option in a dropdown element.

        :param browser_context: The browser context.
        :param index: The index of the dropdown element.
        :param text: The text of the option to select.
        :return: A message indicating the result of the operation.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started selecting the option "
            f"'{text}' in the dropdown with index {index}."
        )
        logger.info(start_message)
        self.on_observation(start_message)
        
        try:
            session = await browser_context.get_session()
            state = session.cached_state

            if index not in state.selector_map:
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                    f"No dropdown found with index {index}. Please verify the index or try alternative actions."
                )
                logger.error(msg)
                return msg

            dom_element = state.selector_map[index]
            initial_pages = len(session.context.pages)

            if dom_element.tag_name.lower() != "select":
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                    f"Element with index {index} is a '{dom_element.tag_name}', not a 'select' element."
                )
                logger.error(msg)
                return msg

            logger.debug(
                f"Attempting to select '{text}' in dropdown with index {index} using XPath: {dom_element.xpath}"
            )
            logger.debug(f"Element attributes: {dom_element.attributes}")
            logger.debug(f"Element tag: {dom_element.tag_name}")

            # Notify observation about the attempt to select the option
            attempt_time = datetime.utcnow().isoformat() + "Z"
            attempt_message = (
                f"[{attempt_time}] Attempting to select option '{text}' in dropdown with index {index}."
            )
            self.on_observation(attempt_message)

            page = await browser_context.get_current_page()
            selector_map = await browser_context.get_selector_map()
            dom_element = selector_map.get(index)

            if not dom_element:
                msg = f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': No dropdown found with index {index}."
                logger.error(msg)
                return msg

            if dom_element.tag_name.lower() != "select":
                msg = (
                    f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                    f"Element with index {index} is a '{dom_element.tag_name}', not a 'select' element."
                )
                logger.error(msg)
                return msg

            xpath = "//" + dom_element.xpath

            frame_index = 0
            for frame in page.frames:
                try:
                    frame_time = datetime.utcnow().isoformat() + "Z"
                    logger.debug(f"[{frame_time}] Trying frame {frame_index} with URL: {frame.url}")

                    # Evaluate the presence of the dropdown in the current frame
                    dropdown_info = await frame.evaluate(
                        """
                        (xpath) => {
                            try {
                                const select = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (!select) return null;
                                if (select.tagName.toLowerCase() !== 'select') {
                                    return {
                                        error: `Found element but it's a ${select.tagName}, not a SELECT`,
                                        found: false
                                    };
                                }
                                return {
                                    id: select.id,
                                    name: select.name,
                                    found: true,
                                    tagName: select.tagName,
                                    optionCount: select.options.length,
                                    currentValue: select.value,
                                    availableOptions: Array.from(select.options).map(o => o.text.trim())
                                };
                            } catch (e) {
                                return {error: e.toString(), found: false};
                            }
                        }
                        """,
                        dom_element.xpath,
                    )

                    if dropdown_info and dropdown_info.get("found"):
                        logger.debug(
                            f"[{datetime.utcnow().isoformat()}Z] Found dropdown in frame {frame_index}: {dropdown_info}"
                        )

                        # Attempt to select the option by its label (text)
                        selected_option_values = (
                            await frame.locator("//" + dom_element.xpath)
                            .nth(0)
                            .select_option(label=text, timeout=5000)
                        )
                        success_time = datetime.utcnow().isoformat() + "Z"
                        msg = (
                            f"[{success_time}] SUCCESS in '{self.name}': "
                            f"Selected option '{text}' with value {selected_option_values} in frame {frame_index}."
                        )
                        logger.info(msg)
                        self.on_observation(msg)

                        # Check if a new tab was opened as a result of the selection
                        if len(session.context.pages) > initial_pages:
                            await browser_context.switch_to_tab(-1)
                            new_tab_time = datetime.utcnow().isoformat() + "Z"
                            new_tab_msg = (
                                f"[{new_tab_time}] INFO in '{self.name}': "
                                f"A new tab was opened and switched to successfully."
                            )
                            msg += f" {new_tab_msg}"
                            logger.info(new_tab_msg)
                            self.on_observation(new_tab_msg)
                        return msg

                    elif dropdown_info and not dropdown_info.get("found"):
                        error_msg = (
                            f"[{datetime.utcnow().isoformat()}Z] ERROR in '{self.name}': "
                            f"{dropdown_info.get('error')}"
                        )
                        logger.error(error_msg)
                        self.on_observation(error_msg)
                        return error_msg

                except Exception as frame_e:
                    frame_error_time = datetime.utcnow().isoformat() + "Z"
                    error_msg = (
                        f"[{frame_error_time}] WARNING in '{self.name}': "
                        f"Frame {frame_index} attempt failed: {str(frame_e)}."
                    )
                    logger.warning(error_msg)
                    self.on_observation(error_msg)
                    logger.debug(f"Frame type: {type(frame)}")
                    logger.debug(f"Frame URL: {frame.url}")

                frame_index += 1

            msg = (
                f"[{datetime.utcnow().isoformat()}Z] INFO in '{self.name}': "
                f"Could not select option '{text}' in any frame."
            )
            logger.info(msg)
            self.on_observation(msg)
            return msg

        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = (
                f"[{error_time}] ERROR in '{self.name}': "
                f"An unexpected error occurred during the selection operation: {str(e)}."
            )
            logger.error(error_msg)
            return error_msg
