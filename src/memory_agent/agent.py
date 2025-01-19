import logging
import re
from typing import Dict, List, Optional

import PIL

from src.memory_agent.tools.save_memory import SaveMemoryTool
from src.memory_agent.tools.do_nothing import DoNothingTool
from src.memory_agent.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    USER_PROMPT_PARSE_CODE_ERROR,
)
from src.utils.llm import LLM
from src.utils.local_python_interpreter import LocalPythonInterpreter
from src.utils.tool import Tool
from src.utils.messages import AssistantMessage, Message, SystemMessage, UserMessage
from src.utils.memory_manager import MemoryManager


class MemoryAgent:
    """
    Agent class that saves memories to long term memory.
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        completion_model_id: str,
        completion_api_base: Optional[str],
        completion_api_key: Optional[str],
        embedding_model_id: str,
        embedding_api_base: Optional[str],
        embedding_api_key: Optional[str],
        **kwargs,
    ):
        logging.debug("Initializing Memory Agent.")
        self.agent_name = self.__class__.__name__

        self.memory_manager = memory_manager

        # Store llm parameters
        self.completion_model_id = completion_model_id
        self.completion_api_base = completion_api_base
        self.completion_api_key = completion_api_key
        self.embedding_model_id = embedding_model_id
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key

        self._initialize_llm()

    def _initialize_llm(self):
        """
        Initialize the LLM instance with the current parameters.
        """
        self.llm = LLM(
            completion_model_id=self.completion_model_id,
            completion_api_base=self.completion_api_base,
            completion_api_key=self.completion_api_key,
            embedding_model_id=self.embedding_model_id,
            embedding_api_base=self.embedding_api_base,
            embedding_api_key=self.embedding_api_key,
            completion_requests_per_minute=5,
        )
        logging.debug("LLM instance initialized.")
    
    def save_memories(self, observations: List[str], observation_images: Dict[str, PIL.Image]):

        try:
            logging.info("📝 Remember observations")
            logging.debug(f"Observations: {observations}")
            observations_string = "\n".join(
                [
                    f"Observation {i}: {observation}"
                    for i, observation in enumerate(self.unprocessed_observations)
                ]
            )

            tools: List[Tool] = [
                SaveMemoryTool(memory_manager=self.memory_manager, images=observation_images),
                DoNothingTool(),
            ]

            tool_descriptions = "\n".join(
                [
                    f"-{tool.name}({tool.inputs}) -> ({tool.output_type}): {tool.description}"
                    for tool in self.tools
                ]
            )

            python_interpreter = LocalPythonInterpreter(tools)

            system_prompt = SYSTEM_PROMPT.format(
                tool_descriptions=tool_descriptions
            )

            user_prompt = USER_PROMPT.format(
                observations = observations_string
            )

            messages = [
                SystemMessage(system_prompt),
                UserMessage(user_prompt),

            ]
            code_blob = self.llm.generate(messages)

            messages_with_code_blob = [
                *messages,
                AssistantMessage(code_blob),
            ]  # Add generated code to messages

            # Parse the generated code with retries
            code = self._parse_with_retries(code_blob, messages_with_code_blob)

            messages_with_parsed_code = [*messages, AssistantMessage(code)]
            # Execute the parsed code with retries
            self._execute_with_retries(code, messages_with_parsed_code, python_interpreter=python_interpreter)

        except Exception as e:
            logging.error(f"❌ Failed to remember observations. {str(e)}.")
        
        # Reset unprocesed observations
        self.unprocessed_observations = []

    def _parse_with_retries(
        self, code_blob: str, messages: List[Message]
    ) -> Optional[str]:
        logging.info("🔧 Parsing code")
        current_messages = messages.copy()
        try:
            return self._parse_code_blob(code_blob)
        except Exception as e:
            error_message = f"An error occurred while parsing code blob. {str(e)}."
            parse_retries = 5
            while parse_retries > 0:
                logging.debug(
                    f"❌ Failed to parse code. Retrying. {error_message}. Attempts left: {parse_retries}"
                )
                try:
                    current_messages.append(
                        UserMessage(error_message)
                    )  # Add error message
                    corrected_code_blob = self.llm.generate(
                        current_messages, stop_sequences=["<end_action>"]
                    )
                    current_messages.append(
                        AssistantMessage(corrected_code_blob)
                    )  # Add corrected code
                    logging.info("🔧 Parsing code")
                    code = self._parse_code_blob(corrected_code_blob)
                    logging.info(f"🩹 Fixed code: {code}.")
                    return code
                except Exception as parse_e:
                    error_message = (
                        f"An error occurred while parsing code blob. {str(parse_e)}."
                    )
                    parse_retries -= 1
                    logging.debug(
                        f"Parsing retry {5 - parse_retries}/5 failed: {error_message}"
                    )
            # If all retries failed
            logging.error(f"❌ Failed to parse code. {error_message}.")
            raise Exception(f"Failed to parse code. {error_message}.")

    def _execute_with_retries(self, code: str, messages: List[Message], python_interpreter: LocalPythonInterpreter):
        logging.info("🧑‍💻 Executing code")

        # We keep track of two different sets of messages
        # current_messages contains all correction attemps and the error messages to inform the model about already tried corrections
        # inner_current_messages contains only the messages for the most recent correction attempt so the parser can focus only on the current corrected code blob        
        current_messages = messages.copy()
        inner_current_messages = messages.copy()
        execute_retries = 5
        while execute_retries > 0:
            _, _, error = python_interpreter(code)

            if error:
                logging.error(f"❌ Failed to execute code: {error}")
                execute_retries -= 1
                if execute_retries == 0:
                    logging.error(
                        f"❌ Failed to execute code after {execute_retries} retries: {error}"
                    )
                    # Pass it to outer code block to notify agent
                    raise Exception(
                        f"Failed to execute code after {execute_retries} retries. {error}"
                    )

                # Prompt LLM to correct the code based on the execution error
                error_message = f"An error occurred during code execution: {error}"
                logging.info("🔧 Attempting to correct code based on execution error.")
                try:
                    current_messages.append(
                        UserMessage(error_message)
                    )  # Add error message
                    inner_current_messages.append(
                        UserMessage(error_message)
                    )  # Add error message of the latest correction attempt
                    corrected_code_blob = self.llm.generate(
                        current_messages, stop_sequences=["<end_action>"]
                    )  # Add corrected code
                    inner_current_messages.append(
                        AssistantMessage(corrected_code_blob)
                    )  # Add corrected code of the latest correction attempt

                    try:
                        # Parse the corrected code with retries
                        code = self._parse_with_retries(
                            corrected_code_blob, inner_current_messages
                        )
                    except:
                        # The corrected code couldn't be parsed try again to fix the original code
                        inner_current_messages = messages.copy() # Reset to original state so it always contains just the messages for the latest correction attempt
                        continue
                except Exception as llm_e:
                    logging.error(f"❌ LLM failed to generate corrected code: {llm_e}")
                    return
            else:
                logging.info(f"✔️ Successfully executed action")
                break

    def _parse_code_blob(self, code_blob: str) -> str:
        """
        Utility to extract Python code from triple-backtick fences, e.g.:

        ```python
        code here
        ```
        """
        try:
            pattern = r"```(?:py|python)?\n(.*?)\n```"
            match = re.search(pattern, code_blob, re.DOTALL)
            if match is None:
                raise Exception(f"No match found for regex pattern {pattern}.")
            code = match.group(1).strip()
            return code

        except Exception as e:
            error_message = USER_PROMPT_PARSE_CODE_ERROR.format(
                code_blob=code_blob,
                error=e,
                pattern=pattern,
                tool_descriptions=self.tool_descriptions,
            )
            raise Exception(error_message)

    def update_completion_model(
        self,
        completion_model_id: str,
        completion_api_base: Optional[str],
        completion_api_key: Optional[str],
        **kwargs,
    ):
        """
        Update the completion model parameters and reinitialize the LLM.
        """
        self.completion_model_id = completion_model_id
        self.completion_api_base = completion_api_base
        self.completion_api_key = completion_api_key
        logging.debug("Completion model parameters updated.")
        self._initialize_llm()

    def update_embedding_model(
        self,
        embedding_model_id: str,
        embedding_api_base: Optional[str],
        embedding_api_key: Optional[str],
        **kwargs,
    ):
        """
        Update the embedding model parameters and reinitialize the LLM.
        """
        self.embedding_model_id = embedding_model_id
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key
        logging.debug("Embedding model parameters updated.")
        self._initialize_llm()
