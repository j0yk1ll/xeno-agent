import logging
import threading
from datetime import datetime
import subprocess
from typing import Callable

from src.utils.tool import Tool

logger = logging.getLogger(__name__)

class UseTerminalTool(Tool):
    name = "use_terminal"
    description = (
        "A tool to execute terminal commands. Use this to interact with the terminal for tasks like file management, running scripts, or checking system status."
    )
    inputs = {
        "command": {
            "type": "string",
            "description": "The terminal command to execute. Ensure correct quoting of arguments and escaping of special symbols. ONLY run commands that are safe and DO NOT modify critical system files.",
        }
    }

    timeout = 360  # Timeout in seconds

    def __init__(self, on_observation: Callable[[str], None]):
        """
        Initialize the UseTerminalTool with a callback function.

        :param on_observation: A callable that takes a single string argument.
        """
        self.on_observation = on_observation
        super().__init__()

    def forward(self, command: str):
        """
        Initiates the terminal command execution in a separate thread.

        :param command: The terminal command to execute.
        """
        initiation_time = datetime.utcnow().isoformat() + "Z"
        initiation_message = (
            f"[{initiation_time}] Initiating the '{self.name}' tool with command: {command}."
        )
        logger.info(initiation_message)

        # Notify observation about initiation
        self.on_observation(initiation_message)

        # Start a new thread to handle the command execution
        thread = threading.Thread(
            target=self._execute_command_thread,
            args=(command,),
            daemon=True  # Daemonize thread to exit with the main program
        )
        thread.start()

    def _execute_command_thread(self, command: str):
        """
        The target method for the thread which handles the terminal command execution.

        :param command: The terminal command to execute.
        """
        execution_start_time = datetime.utcnow().isoformat() + "Z"
        start_message = (
            f"[{execution_start_time}] The '{self.name}' tool started executing the command: {command}."
        )
        logger.info(start_message)
        self.on_observation(start_message)

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            process_id = str(process.pid)
            logger.debug(f"Process ID: {process_id} started for command: {command}")

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return_code = process.returncode

                if return_code == 0:
                    success_time = datetime.utcnow().isoformat() + "Z"
                    success_message = (
                        f"[{success_time}] SUCCESS: Command executed successfully.\n"
                        f"Process ID: {process_id}\n"
                        f"Stdout:\n{stdout}\n"
                        f"Stderr:\n{stderr}"
                    )
                    logger.info(success_message)
                    self.on_observation(success_message)
                else:
                    error_time = datetime.utcnow().isoformat() + "Z"
                    error_message = (
                        f"[{error_time}] ERROR: Command failed with exit code {return_code}.\n"
                        f"Process ID: {process_id}\n"
                        f"Stdout:\n{stdout}\n"
                        f"Stderr:\n{stderr}"
                    )
                    logger.error(error_message)
                    self.on_observation(error_message)

            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                timeout_time = datetime.utcnow().isoformat() + "Z"
                timeout_message = (
                    f"[{timeout_time}] TIMEOUT: Command execution timed out.\n"
                    f"Process ID: {process_id}\n"
                    f"Stdout:\n{stdout}\n"
                    f"Stderr:\n{stderr}"
                )
                logger.warning(timeout_message)
                self.on_observation(timeout_message)

        except Exception as e:
            error_time = datetime.utcnow().isoformat() + "Z"
            error_message = (
                f"[{error_time}] CRITICAL: An unexpected error occurred while executing the command.\n"
                f"Error: {str(e)}"
            )
            logger.error(error_message)
            self.on_observation(error_message)
