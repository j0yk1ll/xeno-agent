import logging
import subprocess
from src.utils.tool import Tool


class TerminalTool(Tool):
    name = "terminal"
    description = "A tool to execute terminal commands. Use this to interact with the terminal for tasks like file management, running scripts, or checking system status."
    inputs = {
        "command": {
            "type": "string",
            "description": "The terminal command to execute. Ensure correct quoting of arguments and escaping of special symbols. ONLY run commands that are safe and DO NOT modify critical system files.",
        }
    }
    output_type = "string"

    timeout = 30  # seconds

    def forward(self, command: str) -> str:
        logging.info(f"🧰 Using tool: {self.name}")
        try:
            logging.debug(f"Executing terminal command with TerminalTool: {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            process_id = str(process.pid)

            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                logging.debug(f"Process ID: {process_id} timed out.")
                return (
                    f"Process ID: {process_id}\n"
                    f"Stdout:\n{stdout}\n"
                    f"Stderr:\n{stderr}\n"
                    "Process timed out and was terminated."
                )

            if return_code == 0:
                logging.debug(f"Process ID: {process_id} completed successfully.")
                return f"Process ID: {process_id}\nStdout:\n{stdout}\nStderr:\n{stderr}"
            else:
                logging.error(
                    f"Process ID: {process_id} failed with exit code {return_code}."
                )
                return (
                    f"Process ID: {process_id}\n"
                    f"Error: Command failed with exit code {return_code}.\n"
                    f"Stderr:\n{stderr}"
                )

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"
