# Copyright 2024 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import wraps
import inspect
import logging
from typing import Callable, Dict, Union, get_type_hints, get_origin, get_args
from transformers.utils.chat_template_utils import _parse_type_hint

AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "any",
    "object",
    "null",
    "array"
]


def convert_type_hints_to_json_schema(func: Callable) -> Dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    properties = {}
    for param_name, param_type in type_hints.items():
        if param_name != "return":
            properties[param_name] = _parse_type_hint(param_type)
            # Determine if the parameter is Optional
            origin = get_origin(param_type)
            args = get_args(param_type)
            is_optional = origin is Union and type(None) in args
            if is_optional:
                properties[param_name]["nullable"] = True
    for param_name in signature.parameters.keys():
        if (
            param_name != "self"
            and signature.parameters[param_name].default != inspect.Parameter.empty
            and param_name not in properties
        ):
            # Handle parameters without type hints but with default values
            properties[param_name] = {"nullable": True}
    return properties


class Tool:
    """
    A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
      description for your tool.
    - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
      or to make a nice space from your tool, and also can be used in the generated description for your tool.

    You can also override the method [`~Tool.setup`] if your tool has an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    """

    name: str
    description: str
    inputs: Dict[str, Dict[str, Union[str, type, bool]]]
    output_type: str

    def __init__(self, *args, **kwargs) -> None:
        self.is_initialized: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.validate_after_init()

    @classmethod
    def validate_after_init(cls) -> None:
        original_init = cls.__init__

        @wraps(original_init)
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.validate_arguments()

        cls.__init__ = new_init

    def validate_arguments(self) -> None:
        required_attributes: Dict[str, type] = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }

        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute '{attr}'.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute '{attr}' should have type '{expected_type.__name__}', "
                    f"got '{type(attr_value).__name__}' instead."
                )

        for input_name, input_content in self.inputs.items():
            if not isinstance(input_content, dict):
                raise TypeError(f"Input '{input_name}' should be a dictionary.")
            if "type" not in input_content or "description" not in input_content:
                raise KeyError(
                    f"Input '{input_name}' should have keys 'type' and 'description', "
                    f"has only {list(input_content.keys())}."
                )
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise ValueError(
                    f"Input '{input_name}': type '{input_content['type']}' is not an authorized value. "
                    f"Must be one of {AUTHORIZED_TYPES}."
                )

        if getattr(self, "output_type", None) not in AUTHORIZED_TYPES:
            raise ValueError(
                f"output_type '{self.output_type}' is not authorized. Must be one of {AUTHORIZED_TYPES}."
            )

        if not hasattr(self, "forward"):
            raise AttributeError("Subclass must implement the 'forward' method.")

        signature = inspect.signature(self.forward)

        expected_params = set(self.inputs.keys())
        actual_params = set(signature.parameters.keys()) - {"self"}
        if actual_params != expected_params:
            raise Exception(
                "Tool's 'forward' method should take 'self' as its first argument, then its next arguments "
                "should match the keys of tool attribute 'inputs'."
            )

        json_schema = convert_type_hints_to_json_schema(self.forward)
        for key, value in self.inputs.items():
            if "nullable" in value:
                if key not in json_schema or "nullable" not in json_schema[key]:
                    raise ValueError(
                        f"Nullable argument '{key}' in inputs should have key 'nullable' set to True in function signature."
                    )
            if key in json_schema and "nullable" in json_schema[key]:
                if "nullable" not in value:
                    raise ValueError(
                        f"Nullable argument '{key}' in function signature should have key 'nullable' set to True in inputs."
                    )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Implement the 'forward' method in your subclass of `Tool`."
        )

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()
        logging.debug(
            f"Calling tool '{self.name}' with args: {args} and kwargs: {kwargs}"
        )
        outputs = self.forward(*args, **kwargs)
        logging.debug(f"Outputs from tool '{self.name}': {outputs}")
        return outputs

    def setup(self) -> None:
        """
        Override this method for any operations that are expensive and need to be executed before using
        your tool, such as loading a large model.
        """
        self.is_initialized = True
