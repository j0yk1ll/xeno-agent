import logging
import re
from typing import List, Optional
from src.conversation_agent.tools.do_nothing import DoNothingTool
from src.conversation_agent.tools.talk import TalkTool
from src.conversation_agent.tools.solve_task import SolveTaskTool
from src.conversation_agent.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_OBSERVATIONS_SUMMARY,
    USER_PROMPT,
    USER_PROMPT_PARSE_CODE_ERROR,
)
from src.utils.llm import LLM
from src.utils.local_python_interpreter import LocalPythonInterpreter
from src.utils.tool import Tool
from src.utils.messages import SystemMessage, UserMessage
from src.utils.threads.task_agent_tread_manager import TaskAgentThreadManager


class ConversationAgent:
    """
    Agent class that solves tasks step by step using a ReAct-like framework.
    It performs cycles of action (LLM-generated code) and observation (execution results).
    """

    def __init__(
        self,
        task_agent_thread_manager: TaskAgentThreadManager,
        completion_model_id: str,
        completion_api_base: Optional[str],
        completion_api_key: Optional[str],
        embedding_model_id: str,
        embedding_api_base: Optional[str],
        embedding_api_key: Optional[str],
        **kwargs
    ):
        logging.debug("Initializing Conversation Agent.")
        self.agent_name = self.__class__.__name__
        
        # Store llm parameters
        self.completion_model_id = completion_model_id
        self.completion_api_base = completion_api_base
        self.completion_api_key = completion_api_key
        self.embedding_model_id = embedding_model_id
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key

        self._initialize_llm()

        self.callback = None

        self.tools: List[Tool] = [
            TalkTool(self._on_talk),
            SolveTaskTool(task_agent_thread_manager, self._on_solve_task_result),
            DoNothingTool(),
        ]

        self.python_interpreter = LocalPythonInterpreter(self.tools)

        self.tool_descriptions = "\n".join(
            [f"-{tool.name}({tool.inputs}) -> ({tool.output_type}): {tool.description}" for tool in self.tools]
        )

        self.system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions
        )

        self.observations = []

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

    def add_observation(self, observation: str):
        self._add_observation(observation)

    def _add_observation(self, observation: str):
        self.observations.append(observation)
        self._on_observation(self.observations[-21:]) # Get the last 21 observations (20 context + 1 most recent observation)

    def _summarize_observations(self, observations: List[str]):
        try:
            logging.info("📝 Summarizing observations")
            logging.debug(f"Observations: {observations}")
            observations_string = "\n".join(
                [
                    f"Observation {i}: {observation}"
                    for i, observation in enumerate(observations)
                ]
            )

            messages = [
                SystemMessage(SYSTEM_PROMPT_OBSERVATIONS_SUMMARY),
                UserMessage(observations_string),
            ]
            llm_output = self.llm.generate(messages)
        except Exception as e:
            logging.error(f"❌ Failed to summarize observations. {str(e)}.")
            raise Exception(f"Error when generating model output:\n{str(e)}")
        
        return llm_output
    
    def _on_observation(self, observations: List[str]):
        try:
            logging.info("🏃 Generating action")

            context_observations = observations[:-1]
            most_recent_observation = observations[-1]

            context = "No context yet."

            if len(context_observations) > 0:
                context = self._summarize_observations(observations=context_observations)

            user_prompt = USER_PROMPT.format(
                observation=most_recent_observation,
                context = context,
            )

            messages = [
                SystemMessage(self.system_prompt),
                UserMessage(user_prompt),
            ]
            llm_output = self.llm.generate(messages)
        except Exception as e:
            logging.error(f"❌ Failed to generate action. {str(e)}.")
            raise Exception(f"Error when generating model output:\n{str(e)}")

        # Parse the generated code with retries
        code = self._parse_with_retries(llm_output)
        if not code:
            return

        # Execute the parsed code with retries
        self._execute_with_retries(code)

    def _parse_with_retries(self, llm_output: str) -> Optional[str]:
        try:
            logging.info("🔧 Parsing code")
            return self._parse_code_blob(llm_output)
        except Exception as e:
            error_message = f"An error occurred while parsing code blob. {str(e)}."
            parse_retries = 5
            while parse_retries > 0:
                logging.warning(f"❌ Failed to parse code. Retrying. {error_message}. Attempts left: {parse_retries}")
                try:
                    message = UserMessage(error_message)
                    llm_output = self.llm.generate(
                        [message], stop_sequences=["<end_action>"]
                    )
                    logging.info("🔧 Parsing code")
                    code = self._parse_code_blob(llm_output)
                    logging.info(f"🩹 Fixed code: {code}.")
                    return code
                except Exception as parse_e:
                    error_message = f"An error occurred while parsing code blob. {str(parse_e)}."
                    parse_retries -= 1
                    logging.warning(f"Parsing retry {5 - parse_retries}/5 failed: {error_message}")
            # If all retries failed
            self._add_observation(f"Failed to parse code. {error_message}.")
            logging.error(f"❌ Failed to parse code. {error_message}.")
            return None

    def _execute_with_retries(self, code: str):
        execute_retries = 5
        while execute_retries > 0:
            if code:
                logging.info("🧑‍💻 Executing code")
                _, _, error = self.python_interpreter(code)

                if error:
                    logging.error(f"❌ Failed to execute code: {error}")
                    execute_retries -= 1
                    if execute_retries == 0:
                        self._add_observation(f"Failed to execute code after retries. {error}")
                        logging.error(f"❌ Failed to execute code after retries: {error}")
                        break

                    # Prompt LLM to correct the code based on the execution error
                    error_message = f"An error occurred during code execution: {error}"
                    logging.info("🔧 Attempting to correct code based on execution error.")
                    try:
                        message = UserMessage(error_message)
                        llm_output = self.llm.generate(
                            [message], stop_sequences=["<end_action>"]
                        )
                        # Parse the corrected code with retries
                        code = self._parse_corrected_code_with_retries(llm_output)
                        if not code:
                            return
                    except Exception as llm_e:
                        logging.error(f"❌ LLM failed to generate corrected code: {llm_e}")
                        return
                else:
                    logging.info(f"✔️ Successfully executed action")
                    break
            else:
                logging.error("❌ No code available to execute.")
                break

    def _parse_corrected_code_with_retries(self, llm_output: str) -> Optional[str]:
        try:
            logging.info("🔧 Parsing corrected code")
            return self._parse_code_blob(llm_output)
        except Exception as parse_e:
            parse_error_message = f"An error occurred while parsing corrected code blob. {str(parse_e)}."
            parse_retries = 5
            while parse_retries > 0:
                logging.warning(f"❌ Failed to parse corrected code. Retrying. {parse_error_message}. Attempts left: {parse_retries}")
                try:
                    message = UserMessage(parse_error_message)
                    llm_output = self.llm.generate(
                        [message], stop_sequences=["<end_action>", "Observation:"]
                    )
                    logging.info("🔧 Parsing corrected code")
                    code = self._parse_code_blob(llm_output)
                    logging.info(f"🩹 Fixed corrected code: {code}.")
                    return code
                except Exception as parse_e2:
                    parse_error_message = f"An error occurred while parsing corrected code blob. {str(parse_e2)}."
                    parse_retries -= 1
                    logging.warning(f"Parsing retry {5 - parse_retries}/5 failed: {parse_error_message}")
            # If all retries failed
            self._add_observation(f"Failed to parse corrected code. {parse_error_message}.")
            logging.error(f"❌ Failed to parse corrected code. {parse_error_message}.")
            return None


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

    def _on_talk(self, utterance: str):
        logging.debug(f"_on_talk with {utterance}")
        if self.callback is not None:
            self.callback(utterance)
        else:
            logging.warning("No callback defined")

    def _on_solve_task_result(self, result: str):
        logging.debug(f"_on_solve_task_result with {result}")
        self._add_observation(result)

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