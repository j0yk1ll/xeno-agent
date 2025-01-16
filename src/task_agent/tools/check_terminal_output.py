import logging
import threading
import psutil
from src.utils.tool import Tool


class CheckTerminalOutputTool(Tool):
    name = "check_terminal_output"
    description = "Check the status of a long-running terminal command. Use this tool to retrieve the latest output of a running process using its process ID."
    inputs = {
        "process_id": {
            "type": "string",
            "description": "The process ID of the running command to check.",
        }
    }
    output_type = "string"

    timeout = 30

    def forward(self, process_id: str) -> str:
        logging.info(f"🧰 Using tool: {self.name}")
        try:
            logging.debug(
                f"Checking terminal output for process ID with CheckTerminalOutputTool: {process_id}"
            )

            # Convert process_id to integer
            try:
                pid = int(process_id)
            except ValueError:
                logging.error(f"Invalid process ID format: {process_id}")
                return f"Invalid process ID format: {process_id}"

            # Retrieve the process using psutil
            try:
                process = psutil.Process(pid)
            except psutil.NoSuchProcess:
                logging.error(f"No process found with ID: {process_id}")
                return f"No process found with ID: {process_id}"
            except psutil.AccessDenied:
                logging.error(f"Access denied to process ID: {process_id}")
                return f"Access denied to process ID: {process_id}"
            except Exception as e:
                logging.error(f"Error retrieving process ID {process_id}: {str(e)}")
                return f"Error retrieving process ID {process_id}: {str(e)}"

            stdout_lines = []
            stderr_lines = []
            timed_out = False

            def monitor_output():
                nonlocal stdout_lines, stderr_lines
                try:
                    # Iterate over the process's stdout and stderr
                    # Note: psutil does not provide direct access to stdout/stderr
                    # You may need to implement inter-process communication or logging
                    # Here, we'll assume that stdout and stderr are being logged elsewhere
                    pass  # Placeholder for actual output retrieval logic
                except Exception as e:
                    logging.error(f"Error monitoring output: {str(e)}")

            thread = threading.Thread(target=monitor_output)
            thread.start()

            thread.join(timeout=self.timeout)

            if thread.is_alive():
                timed_out = True

            # Since psutil doesn't provide direct access to stdout/stderr,
            # you might need to adjust this part based on how you're capturing output
            stdout = "".join(stdout_lines)
            stderr = "".join(stderr_lines)

            if process.is_running():
                logging.debug(
                    f"Process ID: {process_id} is still running.{' Timeout hit.' if timed_out else ''}"
                )
                return (
                    f"Process ID: {process_id}\nStdout:\n{stdout}\nStderr:\n{stderr}\n"
                    f"Process is still running.{' Timeout hit.' if timed_out else ''}"
                )
            else:
                logging.debug(f"Process ID: {process_id} has completed.")
                return (
                    f"Process ID: {process_id}\nStdout:\n{stdout}\nStderr:\n{stderr}\n"
                    f"Process has completed."
                )

        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"
