import asyncio
import logging

from src.utils.browser import Browser
from src.utils.tool import Tool


class UseBrowserTool(Tool):
    """
    A unified tool to manage a browser session and perform tasks within that session.
    This tool handles starting the browser, creating a session, executing a task,
    terminating the session, and stopping the browser.
    """

    name = "use_browser"
    description = (
        "Use a web browser to accomplish tasks such as 'Find the cheapest flight from Hannover to New York in March' or 'Open YouTube'. "
        "This tool starts a browser session, tries to complete the given task, then closes the browser session."
        "It does not persist state between browser sessions. It does return the final result, NOT a website or html."
    )
    inputs = {
        "task": {
            "type": "string",
            "description": "The task to accomplish using the web browser.",
        }
    }
    output_type = "string"

    def __init__(self, browser: Browser, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.browser = browser
        self.loop = loop

    def forward(self, task: str) -> str:
        logging.info(f"🧰 Using tool: {self.name}")
        try:
            # Schedule the coroutine to run in the existing event loop
            future = asyncio.run_coroutine_threadsafe(self._handle_task(task), self.loop)
            # Wait for the result (this will block until the coroutine completes)
            result = future.result()
            return result

        except Exception as e:
            logging.error(f"An unexpected error occurred in forward: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"

    async def _handle_task(self, task: str) -> str:
        """
        Asynchronous handler to manage browser operations.
        """
        try:
            # Create a new session
            session_id = await self.browser.create_session()
            logging.info(f"Created browser session with ID: {session_id}")

            # Use the browser session to perform the task
            logging.debug(
                f"Using browser session ID {session_id} to perform task: {task}"
            )
            result = await self.browser.use(session_id, task)
            logging.info(f"Task result: {result}")

            # Terminate the browser session
            logging.debug(f"Terminating browser session with ID: {session_id}")
            await self.browser.terminate_session(session_id)
            logging.info(f"Terminated browser session with ID: {session_id}")

            return f"Task Result:\n{result}"

        except RuntimeError as runtime_err:
            logging.error(f"Runtime error: {str(runtime_err)}")
            return f"Runtime error: {str(runtime_err)}"
        except ValueError as val_err:
            logging.error(f"Value error: {str(val_err)}")
            return f"Value error: {str(val_err)}"
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"
