import logging

from src.utils.tool import Tool
from src.utils.local_python_interpreter import LocalPythonInterpreter

class PythonInterpreterTool(Tool):
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. Use it to run arbitrary python code."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter. All variables used in this snippet must be defined in this same snippet, else you will get an error. To return the final result use result(value).",
        }
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        self.local_python_interpreter = LocalPythonInterpreter()
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        logging.info(f"🧰 Using tool: {self.name}")
        logging.debug(f"Evaluating Python code with PythonInterpreterTool: {code}")
        output, _, error = self.local_python_interpreter(code)
        logging.debug(f"Output: {output}")

        if error:
            return f"An error occured: {error}"

        return output