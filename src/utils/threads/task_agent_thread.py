import asyncio
import logging
import threading
from typing import Callable, Optional

from src.utils.browser import Browser
from src.task_agent.agent import TaskAgent
from src.task_agent.tools.check_terminal_output import CheckTerminalOutputTool
from src.task_agent.tools.execute_code import ExecuteCodeTool
from src.task_agent.tools.terminal import TerminalTool
from src.task_agent.tools.use_browser import UseBrowserTool


class TaskAgentThread:
    """
    Manages a single TaskAgent in its own asyncio event loop and thread.
    Automatically shuts down after executing the assigned task.
    """

    def __init__(
        self,
        task: str,
        callback: Callable[[str], None],
        on_complete: Callable[["TaskAgentThread"], None],
        completion_model_id: str,
        completion_api_base: Optional[str],
        completion_api_key: Optional[str],
        embedding_model_id: str,
        embedding_api_base: Optional[str],
        embedding_api_key: Optional[str],
        browser_use_model_id: str,
        browser_use_api_base: Optional[str],
        browser_use_api_key: Optional[str],
        **kwargs,
    ):
        """
        Initialize and start the TaskAgentInstance.

        :param config: Configuration dictionary for the TaskAgent.
        :param task: The task to execute.
        :param callback: Callback to invoke with the task result.
        :param on_complete: Callback to notify the manager upon completion.
        """
        self.task = task
        self.callback = callback
        self.on_complete = on_complete
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop_forever, daemon=True)
        self.thread.start()

        # Initialize the agent asynchronously on the loop
        init_future = asyncio.run_coroutine_threadsafe(
            self._init_agent(
                completion_model_id,
                completion_api_base,
                completion_api_key,
                embedding_model_id,
                embedding_api_base,
                embedding_api_key,
                browser_use_model_id,
                browser_use_api_base,
                browser_use_api_key,
            ),
            self.loop,
        )
        try:
            init_future.result()  # Block until agent is fully initialized.
        except Exception as e:
            logging.error(f"Failed to initialize TaskAgentInstance: {e}")
            self.shutdown()
            raise

        # Execute the task
        self.execute_task()

    def _run_loop_forever(self):
        """Run the event loop forever in the background thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_agent(
        self,
        completion_model_id: str,
        completion_api_base: Optional[str],
        completion_api_key: Optional[str],
        embedding_model_id: str,
        embedding_api_base: Optional[str],
        embedding_api_key: Optional[str],
        browser_use_model_id: str,
        browser_use_api_base: Optional[str],
        browser_use_api_key: Optional[str],
    ):
        """(Async) Initialize the tools, browser, and agent."""
        # Unpack configuration

        # Create a browser instance (if required)
        self.browser = Browser(
            model_id=browser_use_model_id,
            api_base=browser_use_api_base,
            api_key=browser_use_api_key,
        )
        await self.browser.start()

        # Define available tools
        tools = [
            ExecuteCodeTool(),
            UseBrowserTool(browser=self.browser, loop=self.loop),
            TerminalTool(),
            CheckTerminalOutputTool(),
        ]


        # Create the agent
        self.agent = TaskAgent(
            completion_model_id=completion_model_id,
            completion_api_base=completion_api_base,
            completion_api_key=completion_api_key,
            embedding_model_id=embedding_model_id,
            embedding_api_base=embedding_api_base,
            embedding_api_key=embedding_api_key,
            tools=tools,
            planning_interval=1,
            compression_interval=5,
        )

    def execute_task(self):
        """Schedule the task to run in the event loop."""
        asyncio.run_coroutine_threadsafe(self._execute_task(), self.loop)

    async def _execute_task(self):
        """Coroutine to execute the task."""
        try:
            # Execute the synchronous agent.run in a thread-safe manner
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.agent.run, self.task)
            self.callback(result)
        except Exception as e:
            logging.error(f"Error executing task '{self.task}': {e}")
            self.callback(f"Error: {str(e)}")
        finally:
            self.shutdown()
            self.on_complete(self)

    def shutdown(self):
        """Gracefully stop the browser and the event loop/thread."""
        if hasattr(self, "browser"):
            try:
                awaitable = self.browser.stop()
                future = asyncio.run_coroutine_threadsafe(awaitable, self.loop)
                future.result()
            except Exception as e:
                logging.error(f"Error shutting down browser: {e}")

        self.loop.call_soon_threadsafe(self.loop.stop())
        self.thread.join()
