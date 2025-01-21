import logging
import threading
import asyncio
from typing import Callable
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.context import BrowserContext
from main_content_extractor import MainContentExtractor

logger = logging.getLogger(__name__)

class ExtractPageContentTool(Tool):
    name = "extract_page_content"
    description = "Extract the main content of the current page as text or markdown with links."
    inputs = {
        "browser_context": {
            "type": "object",
            "description": "The browser context."
        },
        "include_links": {
            "type": "boolean",
            "description": "Whether to include links in the extracted content."
        },
    }

    def __init__(self, on_observation: Callable[[str], None], *args, **kwargs):
        """
        Initializes the ExtractPageContentTool with a callback function.

        Args:
            on_observation (Callable[[str], None]): A callback function to handle the result.
        """
        self.on_observation = on_observation
        super().__init__(*args, **kwargs)

    def forward(self, browser_context: BrowserContext, include_links: bool):
        """
        Initiates the extraction process in a separate thread.

        Args:
            include_links (bool): Whether to include links in the extracted content.
            browser_context (BrowserContext): The browser context.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with parameters - "
            f"include_links: {include_links}."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous extraction
        thread = threading.Thread(
            target=self._extract_content_thread,
            args=(browser_context, include_links),
            daemon=True  # Daemonize thread to ensure it exits with the main program
        )
        thread.start()

    def _extract_content_thread(self, browser_context: BrowserContext, include_links: bool):
        """
        The target method for the thread which handles the asynchronous extraction operation.

        Args:
            browser_context (BrowserContext): The browser context.
            include_links (bool): Whether to include links in the extracted content.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous extraction coroutine
            result = loop.run_until_complete(
                self._async_extract(browser_context, include_links)
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

    async def _async_extract(self, browser_context: BrowserContext, include_links: bool) -> str:
        """
        Asynchronous method to extract the page content.

        Args:
            browser_context (BrowserContext): The browser context.
            include_links (bool): Whether to include links in the extracted content.

        Returns:
            str: A message indicating the result of the extraction.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] The '{self.name}' tool has started extracting the page content "
            f"with include_links={include_links}."
        )
        logger.info(start_message)
        self.on_observation(start_message)
        
        try:
            page = await browser_context.get_current_page()
            output_format = 'markdown' if include_links else 'text'
            html_content = await page.content()
            content = MainContentExtractor.extract(html=html_content, output_format=output_format)
            success_time = datetime.utcnow().isoformat() + "Z"
            success_message = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f"Successfully extracted page content as {output_format}."
                f"\n{content}\n"
            )
            logger.info(success_message)
            return success_message
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': {str(e)}."
            logger.error(error_msg)
            raise
