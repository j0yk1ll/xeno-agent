import ast
import sys
import io
import types
import inspect
from typing import List, Optional

from src.utils.tool import Tool


class ResultException(Exception):
    """An internal exception used to capture result(value) calls."""

    def __init__(self, value):
        self.value = value


class LogException(Exception):
    """An internal exception used to capture log messages."""

    def __init__(self, message):
        self.message = message


class LocalPythonInterpreter:
    def __init__(self, tools: Optional[List[Tool]]=None):
        """
        :param tools: An optional list of `Tool` objects to be injected into the interpreter environment.
        """
        # Store tools (if any)
        self._tools = tools or []

        # Initialize globals with built-in globals, custom functions, and (optionally) the provided tools
        self._initialize_globals()

        # A set to keep track of imported modules (by name).
        self.imported_modules = set()

    def _initialize_globals(self):
        # Global namespace. We attach __builtins__ for normal Python usage.
        self._globals = {
            "__builtins__": __builtins__,
        }

        # Provide a custom function `result(value)` that raises an internal exception to "return" a value.
        def result(value):
            raise ResultException(value)

        # Provide a custom function `log(message)` that appends messages to a log list.
        self._logs = []

        def log(message):
            self._logs.append(message)

        # Inject them into the environment
        self._globals["result"] = result
        self._globals["log"] = log

        # Inject the provided tools into the environment, keyed by their `Tool.name`.
        for tool in self._tools:
            # e.g. if tool.name = "text_classifier", you can call it in Python as text_classifier(...)
            self._globals[tool.name] = tool

    def _capture_imports(self, code: str):
        """
        Parse the code for import statements and record imported modules.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If the code is invalid Python, we won't capture imports.
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # e.g., "import math" -> alias.name = 'math'
                    self.imported_modules.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                # e.g., "from math import sqrt" -> node.module = 'math'
                if node.module is not None:
                    self.imported_modules.add(node.module)

    def __call__(self, code: str):
        """
        Execute the given code in our interpreter, capturing stdout and errors.
        Returns (result, logs, error):
          - result: The value from `result(...)` if called, or None otherwise.
          - logs: A list of messages from `log(...)` calls.
          - error: Any error if an exception occurred, or None if no error.
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = stdout_buffer, stderr_buffer

        result_value = None
        logs = []
        error = None

        # Clear previous logs
        self._logs.clear()

        # Capture imports in this code snippet
        self._capture_imports(code)

        try:
            exec(code, self._globals, self._globals)
        except ResultException as re:
            result_value = re.value
        except Exception as e:
            error = str(e)
        finally:
            # Capture logs after execution
            logs = self._logs.copy()
            sys.stdout, sys.stderr = old_stdout, old_stderr

        return result_value, logs, error

    # -----------------------------------------
    # HELPER FUNCTIONS FOR JSON-SERIALIZABLE PROPERTIES
    # -----------------------------------------

    def _module_representation(self, module: types.ModuleType) -> str:
        """
        Return a string that includes the module's name and whether it's built-in or custom.
        Example: "'math' (built-in)" or "'utils' (custom)"
        """
        name = getattr(module, "__name__", "unknown")
        # If __file__ is absent or None, it's likely built-in
        if not hasattr(module, "__file__") or module.__file__ is None:
            return f"'{name}' (built-in)"
        else:
            return f"'{name}' (custom)"

    def _function_signature(self, func: types.FunctionType) -> str:
        """
        Return a string representation of the function's signature.
        Example: "sum(a, b) -> int" if type hints are present,
        or "sum(a, b)" if no type hints are available.
        """
        sig = inspect.signature(func)
        return f"{func.__name__}{sig}"

    def _class_representation(self, cls: type) -> dict:
        """
        Return a JSON-serializable object with keys:
          - "name": Name of the class
          - "variables": List of attribute names that are not callables and not dunder
          - "methods": A dict, whose keys are method names and values are method signatures
        """
        class_dict = {"name": cls.__name__, "variables": [], "methods": {}}

        for attr_name, attr_value in cls.__dict__.items():
            # Skip dunder attributes
            if attr_name.startswith("__") and attr_name.endswith("__"):
                continue
            # If it's a function, record it in "methods"
            if isinstance(attr_value, types.FunctionType):
                class_dict["methods"][attr_name] = self._function_signature(attr_value)
            # Otherwise, if not a callable, treat it as a variable
            elif not callable(attr_value):
                class_dict["variables"].append(attr_name)

        return class_dict

    # -----------------------------------------
    # PROPERTIES
    # -----------------------------------------
    @property
    def variables(self):
        """
        Return only normal variables (exclude modules, functions, classes, hidden items).
        """
        filtered = {}
        for k, v in self._globals.items():
            # Skip Python internals and our custom "result" and "log" functions
            if (k.startswith("__") and k.endswith("__")) or k in {"result", "log"}:
                continue
            # Exclude modules
            if isinstance(v, types.ModuleType):
                continue
            # Exclude functions (including built-in or user-defined)
            if isinstance(v, types.FunctionType):
                continue
            # Exclude classes
            if isinstance(v, type):
                continue
            # Exclude tools
            if isinstance(v, Tool):
                continue
            else:
                filtered[k] = v
        return filtered

    @property
    def modules(self):
        """
        Return a list of string representations of modules in the interpreter's namespace.
        e.g. ["'math' (built-in)", "'utils' (custom)", ... ]
        """
        module_list = []
        for k, v in self._globals.items():
            if isinstance(v, types.ModuleType) and not (
                k.startswith("__") and k.endswith("__")
            ):
                module_list.append(self._module_representation(v))
        return module_list

    @property
    def functions(self):
        """
        Return a list of function signatures
        e.g. ["sum(a, b)", "sub(a, b)"]
        """
        funcs = []
        for k, v in self._globals.items():
            if (
                isinstance(v, types.FunctionType)
                and k not in {"result", "log"}
                and not (k.startswith("__") and k.endswith("__"))
            ):
                funcs.append(self._function_signature(v))
        return funcs

    @property
    def classes(self):
        """
        Return a list of JSON-serializable objects describing each class.
        Each object has the keys:
          - "name"
          - "variables"
          - "methods"

        e.g. [
            {
                "name": "Calculator",
                "variables": [],
                "methods": {
                    "sub": "sub(a, b)"
                }
            }
        ]
        """
        class_list = []
        for k, v in self._globals.items():
            if isinstance(v, type) and not (k.startswith("__") and k.endswith("__")):
                class_list.append(self._class_representation(v))
        return class_list

    @property
    def state(self):
        """
        Return a JSON-serializable object reflecting the state of the local python interpreter.
        """
        return {
            "modules": self.modules,
            "functions": self.functions,
            "classes": self.classes,
            "variables": self.variables
        }
    # -----------------------------------------

    # METHODS

    # -----------------------------------------

    def reset(self):
        """
        Resets the interpreter to its initial state by clearing all variables, functions, classes, and imported modules.
        It reinitializes the global namespace and clears the imported_modules set.
        """
        self._initialize_globals()
        self.imported_modules.clear()

if __name__ == "__main__":
    interpreter = LocalPythonInterpreter()

    code_snippet = """
import math

x = 42
y = math.sqrt(16)

def sum(a, b):
    return a + b

class Calculator:
    def sub(self, a, b):
        return a - b

    def div(self, a, b):
        return a / b

calc = Calculator()

log("Calculator initialized.")
log(f"x = {x}")
log(f"y = {y}")

result(sum(9, 1))
    """

    output, logs, error = interpreter(code_snippet)

    print("\n*** 1 ***")
    print("Output:", output)
    print("Logs:", logs)
    print("Error:", error)
    print("Variables:", interpreter.variables)

    print("Modules:", interpreter.modules)

    print("Functions:", interpreter.functions)

    print("Classes:", interpreter.classes)

    code_snippet = """
log("Calling sub method.")
result(calc.sub(9, 1))
    """

    output, logs, error = interpreter(code_snippet)

    print("\n*** 2 ***")
    print("Output:", output)
    print("Logs:", logs)
    print("Error:", error)
    print("Variables:", interpreter.variables)

    print("Modules:", interpreter.modules)

    print("Functions:", interpreter.functions)

    print("Classes:", interpreter.classes)

    interpreter.reset()

    code_snippet = """
result(calc.sub(9, 1))
    """

    output, logs, error = interpreter(code_snippet)

    print("\n*** 3 ***")
    print("Output:", output)
    print("Logs:", logs)
    print("Error:", error)
    print("Variables:", interpreter.variables)

    print("Modules:", interpreter.modules)

    print("Functions:", interpreter.functions)

    print("Classes:", interpreter.classes)
