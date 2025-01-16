import logging
from typing import Callable

from src.utils.tool import Tool
from src.utils.threads.task_agent_tread_manager import TaskAgentThreadManager


class SolveTaskTool(Tool):
    name: str = "solve_task"
    description: str = "Use this tool to solve a task given by the user. Returns the result."
    inputs = {
        "task": {
            "type": "string",
            "description": "The instruction or task to solve.",
        }
    }
    output_type: str = "string"

    def __init__(self, task_agent_thread_manager: TaskAgentThreadManager, on_result: Callable[[str], None]):
        """
        :param task_agent_manager: An instance of TaskAgentManager, which manages the background asyncio loop/thread.
        :param on_result: A callable that will be invoked with the result once the agent has processed the task.
        """
        self.task_agent_thread_manager = task_agent_thread_manager
        self.on_result = on_result
        super().__init__()

    def forward(self, task: str) -> str:
        """
        1. Submits the task to the AgentManager (which uses an Agent) asynchronously.
        2. Returns immediately without waiting for the result.
        3. The result is passed back via the on_result callback when ready.

        :param task: The task to be processed by the Task Agent.
        :return: A placeholder string indicating that the task is being processed.
        """
        logging.debug(f"[SolveTaskTool] Forward called with task: {task}")

        # Schedule the agent run asynchronously with the on_result callback
        self.task_agent_thread_manager.run_task_async(task, self.on_result)

        logging.debug("[SolveTaskTool] Task submitted asynchronously.")

        # Return a placeholder response immediately
        return "Task is being processed. You will receive the result shortly."
