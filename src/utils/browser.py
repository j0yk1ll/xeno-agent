import logging
import os
import uuid
import asyncio
from typing import Dict, Optional
from dotenv import load_dotenv

from browser_use import (
    Agent as BrowserUseAgent,
    Browser as BrowserUseBrowser,
    BrowserConfig as BrowserUseConfig,
)
from browser_use.browser.context import BrowserContext
from langchain_community.chat_models import ChatLiteLLM

# Load environment variables
load_dotenv(override=True)


class Browser:
    """
    A class-based interface to manage browser sessions without using an API.
    It supports creating and terminating sessions and running queries in a session.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str],
        api_key: Optional[str],
        headless: bool = False,
    ):
        """
        :param headless: Whether to run the browser in headless mode.
        """
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        self.headless = headless
        self.browser: Optional[Browser] = None
        self.sessions: Dict[str, BrowserContext] = {}
        self.sessions_lock = asyncio.Lock()

    async def start(self) -> None:
        """
        Initialize the Browser instance.
        Call this once before creating or using any sessions.
        """
        if self.browser is not None:
            # Browser is already started.
            return

        try:
            browser_config = BrowserUseConfig(headless=self.headless)
            self.browser = BrowserUseBrowser(config=browser_config)
            logging.debug("Browser launched successfully.")
        except Exception as e:
            logging.debug(f"Failed to launch browser: {e}")
            raise e

    async def stop(self) -> None:
        """
        Close all active browser contexts and the Browser instance.
        Call this once you're done with all sessions.
        """
        if not self.browser:
            return  # Browser is already stopped or was never started

        async with self.sessions_lock:
            session_ids = list(self.sessions.keys())
            for session_id in session_ids:
                context = self.sessions.pop(session_id)
                try:
                    await context.close()
                    logging.debug(f"Browser context {session_id} closed.")
                except Exception as e:
                    logging.debug(f"Failed to close browser context {session_id}: {e}")

        await self.browser.close()
        self.browser = None
        logging.debug("Browser closed successfully.")

    async def create_session(self) -> str:
        """
        Create a new browser session (context) and return the session ID.
        :return: The session ID string.
        """
        if not self.browser:
            raise RuntimeError("Browser must be started before creating sessions.")

        session_id = str(uuid.uuid4())

        try:
            # Create a new browser context
            context = await self.browser.new_context()
        except Exception as e:
            raise RuntimeError(f"Failed to create browser session: {str(e)}")

        async with self.sessions_lock:
            self.sessions[session_id] = context

        logging.debug(f"Session {session_id} created.")
        return session_id

    async def terminate_session(self, session_id: str) -> None:
        """
        Terminate and remove the specified browser session.
        :param session_id: The ID of the session to terminate.
        """
        async with self.sessions_lock:
            context = self.sessions.pop(session_id, None)

        if not context:
            raise ValueError(f"Session {session_id} not found")

        try:
            await context.close()
            logging.debug(f"Session {session_id} terminated successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to close browser session {session_id}: {str(e)}"
            )

    async def use(self, session_id: str, task: str) -> str:
        """
        Execute a task within the specified browser session.
        :param session_id: The ID of the session to use.
        :param task: The text describing the task to execute.
        :return: The result string from the executed task.
        """
        if not task:
            raise ValueError("Task cannot be empty")

        async with self.sessions_lock:
            context = self.sessions.get(session_id)

        if not context:
            raise ValueError(f"Session {session_id} not found")

        # Initialize the language model (assuming it's lightweight)
        llm = ChatLiteLLM(
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
        )

        # Create an Agent using the specified browser context
        agent = BrowserUseAgent(
            task=task,
            llm=llm,
            browser_context=context,  # Use the specific browser session
            generate_gif=False
        )

        try:
            result = await agent.run()
            logging.debug(f"Browser Agent Result: {result}")
            return result
        except Exception as e:
            logging.debug(f"An error occured during Browser.use: {str(e)}")
            raise RuntimeError(str(e))

    async def list_sessions(self) -> Dict[str, str]:
        """
        List all active sessions.
        :return: A dict of {session_id: "active"} for each active session.
        """
        async with self.sessions_lock:
            return {session_id: "active" for session_id in self.sessions}


# Example usage
async def main():
    import os
    browser = Browser(
        model_id="",
        api_base="",
        api_key="",
    )
    await browser.start()

    # Create a new session
    session_id = await browser.create_session()
    logging.debug("Created session:", session_id)

    # Use the session to execute a task
    try:
        result = await browser.use(
            session_id, "Open the example website and summarize its content."
        )
        logging.debug("Result:", result)
    except Exception as e:
        logging.debug("Error executing task:", e)

    # Terminate the session
    await browser.terminate_session(session_id)

    # Stop the browser
    await browser.stop()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
