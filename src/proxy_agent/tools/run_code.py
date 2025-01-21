import logging
import threading
import asyncio
from typing import Callable
from datetime import datetime

from src.utils.tool import Tool
from src.utils.local_python_interpreter import LocalPythonInterpreter

logger = logging.getLogger(__name__)

class RunCodeTool(Tool):
    name = "run_code"
    description = "Use it to run Python code."
    inputs = {
        "code": {
            "type": "string",
            "description": (
                "The Python code to execute. All variables used in this snippet must be defined in this same snippet, "
                "else you will get an error. To return the final result use result(value)."
            ),
        }
    }

    def __init__(self, on_observation: Callable[[str], None]):
        """
        Initialize the ExecuteCodeTool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        self.local_python_interpreter = LocalPythonInterpreter()
        super().__init__()

    def forward(self, code: str):
        """
        Initiates the code execution in a separate thread.

        :param code: The Python code to execute.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with provided code."
        )
        logger.info(initiation_message)
        logger.debug(f"Code to execute:\n{code}")

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the asynchronous code execution
        thread = threading.Thread(
            target=self._execute_code_thread,
            args=(code,),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _execute_code_thread(self, code: str):
        """
        The target method for the thread which handles the asynchronous code execution.

        :param code: The Python code to execute.
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Run the asynchronous execute method
            result = loop.run_until_complete(
                self._async_execute(code)
            )
            # Invoke the callback with the result
            self.on_observation(result)
        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': An unexpected error occurred during code execution: {str(e)}."
            logger.error(error_msg)
            self.on_observation(error_msg)
        finally:
            # Close the event loop
            loop.close()

    async def _async_execute(self, code: str) -> str:
        """
        The asynchronous method that performs the actual code execution.

        :param code: The Python code to execute.
        :return: A message indicating the result of the code execution.
        """
        execution_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = f"[{execution_start_time}] Starting execution of provided Python code."
        logger.info(start_message)
        self.on_observation(start_message)
        
        try:
            # Execute the code using the local Python interpreter
            logger.debug("Passing code to LocalPythonInterpreter for execution.")
            output, _, error = self.local_python_interpreter(code)
            logger.debug(f"Execution output: {output}")
            logger.debug(f"Execution error: {error}")

            if error:
                error_time = datetime.utcnow().isoformat() + "Z"
                error_msg = f"[{error_time}] ERROR in '{self.name}': An error occurred during code execution: {error}."
                logger.error(error_msg)
                return error_msg

            success_time = datetime.utcnow().isoformat() + "Z"
            success_msg = f"[{success_time}] SUCCESS in '{self.name}': Code executed successfully. Output:\n{output}"
            logger.info(success_msg)
            return success_msg

        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_msg = f"[{error_time}] ERROR in '{self.name}': Exception during code execution: {str(e)}."
            logger.error(error_msg)
            return error_msg
