import logging
import threading
import asyncio
from typing import Callable
from datetime import datetime

from src.utils.tool import Tool
from src.utils.browser.browser import Browser
from src.utils.browser.context import BrowserContext

# Configure the logger for this module
logger = logging.getLogger(__name__)

class CreateBrowserContextTool(Tool):
    name = "create_browser_context"
    description = "Create a new browser context."
    inputs = {}

    def __init__(self, browser: Browser, on_observation: Callable[[str], None]):
        """
        Initialize the CreateBrowserContextTool with a browser instance and an observation callback.

        :param browser: An instance of the Browser class.
        :param on_observation: A callable that takes a single string argument for observations.
        """
        self.browser = browser
        self.on_observation = on_observation
        super().__init__()

    def forward(self):
        """
        Initiates the creation of a new browser context in a separate thread.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool to create a new browser context."
        )
        logger.info(initiation_message)
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous context creation
        thread = threading.Thread(
            target=self._create_context_thread,
            daemon=True  # Daemonize thread to ensure it exits with the main program
        )
        thread.start()

    def _create_context_thread(self):
        """
        The target method for the thread which handles the asynchronous creation of a browser context.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.debug("Event loop created for context creation thread.")
            # Run the asynchronous context creation
            browser_context = loop.run_until_complete(self._async_create_context())
            success_time = datetime.utcnow().isoformat() + "Z"
            success_message = (
                f"[{success_time}] SUCCESS in '{self.name}': "
                f"Successfully created a new browser context with ID: {browser_context.id}."
            )
            logger.info(success_message)
            self.on_observation(success_message)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_message = (
                f"[{error_time}] ERROR in '{self.name}': Failed to create a new browser context. "
                f"Error: {str(e)}."
            )
            logger.error(error_message)
            self.on_observation(error_message)
        finally:
            # Close the event loop
            loop.close()
            logger.debug("Event loop closed for context creation thread.")

    async def _async_create_context(self) -> BrowserContext:
        """
        Asynchronously creates a new browser context.

        :return: An instance of BrowserContext representing the new context.
        """
        operation_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{operation_start_time}] '{self.name}' tool has started creating a new browser context."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            # Asynchronously create a new browser context
            browser_context = await self.browser.new_context()
            creation_time = datetime.utcnow().isoformat() + "Z"
            creation_message = (
                f"[{creation_time}] '{self.name}' tool successfully created a new browser context "
                f"with ID: {browser_context.id}."
            )
            logger.info(creation_message)
            self.on_observation(creation_message)
            return browser_context
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_message = (
                f"[{error_time}] ERROR in '{self.name}': Exception occurred while creating browser context. "
                f"Error: {str(e)}."
            )
            logger.error(error_message)
            self.on_observation(error_message)
            raise  # Re-raise exception to be caught in the thread

    def __repr__(self):
        return f"<CreateBrowserContextTool(name={self.name})>"
