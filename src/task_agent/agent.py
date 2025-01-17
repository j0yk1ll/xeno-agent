import json
import re
import logging
from typing import List, Optional, Union
import types
import inspect


from src.utils.local_python_interpreter import LocalPythonInterpreter

from src.task_agent.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_FACTS,
    USER_PROMPT_FACTS,
    SYSTEM_PROMPT_FACTS_UPDATE,
    USER_PROMPT_FACTS_UPDATE,
    SYSTEM_PROMPT_PLAN,
    USER_PROMPT_PLAN,
    SYSTEM_PROMPT_PLAN_UPDATE,
    USER_PROMPT_PLAN_UPDATE,
    USER_PROMPT_PARSE_CODE_ERROR,
    SYSTEM_PROMPT_SUMMARIZE_MESSAGES,
)
from src.utils.tool import Tool
from src.task_agent.messages import (
    FinalAnswerMessage,
    ErrorMessage,
    PlanMessage,
    StepResultMessage,
    TaskMessage,
)
from src.task_agent.output_types import handle_agent_output_types
from src.utils.llm import LLM
from src.utils.messages import (
    AssistantMessage,
    Message,
    MessageRole,
    SystemMessage,
    UserMessage,
)


class MessageLedger:
    def __init__(self, llm: LLM, messages: Optional[List[Message]] = None):
        self.llm = llm
        self.messages = messages if messages else []
        logging.debug("Initialized MessageLedger.")

    def add(self, message: Message):
        self.messages.append(message)

    def reset(self):
        self.messages = []

    def length(self):
        return len(self.messages)

    def copy(self):
        return self.messages.copy()

    def summarize(self):
        logging.info("📝 Summarizing messages")
        original_messages = self.messages.copy()

        # Exclude system prompts
        filtered_original_messages = [
            msg for msg in original_messages if msg.role != MessageRole.SYSTEM.value
        ]

        messages_string = "\n".join(
            [f"{m.role}: {m.content}" for m in filtered_original_messages]
        )

        messages = [
            SystemMessage(SYSTEM_PROMPT_SUMMARIZE_MESSAGES),
            UserMessage(messages_string),
        ]

        try:
            summary = self.llm.generate(messages)
            logging.debug(f"Summarized messages: {summary}")
            return summary
        except Exception as e:
            logging.debug(f"❌ Failed to summarize messages. {str(e)}")


class TaskAgent:
    """
    Agent class that solves tasks step by step using a ReAct-like framework.
    It performs cycles of action (LLM-generated code) and observation (execution results).
    """

    def __init__(
        self,
        completion_model_id: str,
        completion_api_base: Optional[str],
        completion_api_key: Optional[str],
        embedding_model_id: str,
        embedding_api_base: Optional[str],
        embedding_api_key: Optional[str],
        tools: Union[List[Tool]],
        max_steps: Optional[int] = None,
        planning_interval: Optional[int] = None,
        compression_interval: Optional[int] = None,
        **kwargs,
    ):
        logging.debug("Initializing TaskAgent.")
        self.agent_name = self.__class__.__name__

        # Store llm parameters
        self.completion_model_id = completion_model_id
        self.completion_api_base = completion_api_base
        self.completion_api_key = completion_api_key
        self.embedding_model_id = embedding_model_id
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key

        self._initialize_llm()

        self.max_steps = max_steps if max_steps else float("inf")
        self.planning_interval = planning_interval
        self.compression_interval = compression_interval

        self.n_steps = 0

        self.tools: List[Tool] = tools

        self.python_interpreter = LocalPythonInterpreter(self.tools)

        self.tool_descriptions = "\n".join(
            [
                f"-{tool.name}({tool.inputs}) -> ({tool.output_type}): {tool.description}"
                for tool in self.tools
            ]
        )

        self.system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions,
        )

        self.facts = None
        self.plan = None

        self.message_ledger = MessageLedger(self.llm, [])

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

    def run(self, task: str):
        """
        Main entry point to start the agent on a given task.
        """

        # Reset message_ledger
        self.message_ledger.reset()

        return self._run(task)

    def _run(self, task: str):
        logging.info(f"🚀 Starting task: {task}")
        final_answer = None

        while final_answer is None and self.n_steps < self.max_steps:
            # Compress messages
            if (
                self.compression_interval is not None
                and self.n_steps % self.compression_interval == 0
                and self.n_steps != 0
            ):
                logging.debug("Compressing messages.")
                summary = self.message_ledger.summarize()

                self.message_ledger.messages = [AssistantMessage(summary)]

            # Possibly plan
            if (
                self.planning_interval is not None
                and self.n_steps % self.planning_interval == 0
            ):
                logging.debug("Performing planning step.")
                self._planning_step(
                    task, is_first_step=(self.n_steps == 0), step=self.n_steps
                )

            try:
                final_answer = self._step(task)
                logging.debug(f"Step returned: {final_answer}")
            except Exception as e:
                logging.debug(f"Exception during step execution. {str(e)}")
                break

            self.n_steps += 1

        if final_answer is None:
            if self.n_steps == self.max_steps:
                error_message = "Reached maximum number of steps."
                self.message_ledger.add(ErrorMessage(error_message))
                final_answer = self._generate_final_answer(task)
                logging.info("❌ Failed to complete task in maximum steps.")
            else:
                logging.info("❌ Failed to complete task.")
        else:
            logging.info("✅ Task completed successfully.")

        logging.info(f"🏆 Result: {final_answer}")

        return handle_agent_output_types(final_answer)

    def _step(self, task: str):
        logging.info(f"📍 Step {self.n_steps}")

        ledger_messages = self.message_ledger.copy()

        try:
            logging.info("🏃 Generating action")
            messages = [
                SystemMessage(self.system_prompt),
                TaskMessage(task),
                *ledger_messages,
            ]
            code_blob = self.llm.generate(messages, stop_sequences=["<end_code>"])
        except Exception as e:
            logging.error(f"❌ Failed to generate action. {str(e)}.")
            raise Exception(f"Error when generating model output:\n{str(e)}")

        # Prepare messages for parsing
        messages_with_code_blob = [*messages, AssistantMessage(code_blob)]

        # Parse code with retry mechanism
        try:
            code = self._parse_with_retries(code_blob, messages_with_code_blob)
        except:
            return None

        # Prepare messages for execution
        messages_with_parsed_code = [*messages, AssistantMessage(code)]

        # Execute code with retry mechanism
        return self._execute_with_retries(code, messages_with_parsed_code)


    def _parse_with_retries(
        self, code_blob: str, messages: List[Message], retries: int = 5
    ) -> str:
        """
        Parses the code with a retry mechanism, providing context messages
        (including the system and user messages) to the LLM on each retry.
        Raises an exception if it cannot parse after all retries.
        """
        current_messages = messages.copy()

        try:
            logging.info("🔧 Parsing code")
            return self._parse_code_blob(code_blob)
        except Exception as e:
            error = str(e)
            logging.warning(f"❌ Failed to parse code. {error}")
            parse_retries = retries

            while parse_retries > 0:
                parse_retries -= 1
                # Inform the model about the parsing error
                error_message = f"An error occurred while parsing code blob. {error}."
                current_messages.append(UserMessage(error_message))
                try:
                    # Ask LLM to correct the code
                    corrected_code = self.llm.generate(
                        current_messages, stop_sequences=["<end_code>"]
                    )
                    # Add the corrected code as an Assistant message
                    current_messages.append(AssistantMessage(corrected_code))

                    logging.info("🔧 Parsing corrected code")
                    code = self._parse_code_blob(corrected_code)
                    logging.info(f"🩹 Fixed code: {code}")
                    return code
                except Exception as e2:
                    error = str(e2)
                    logging.warning(
                        f"❌ Parsing retry failed: {error}. "
                        f"Attempts left: {parse_retries}"
                    )

            # After all retries exhausted, raise an exception
            self.message_ledger.add(ErrorMessage(error_message))
            logging.error(f"❌ Failed to parse code after {retries} retries: {error}")
            raise Exception(error)


    def _execute_with_retries(
        self, code: str, messages: List[Message], retries: int = 5
    ) -> Optional[str]:
        """
        Executes the code with a retry mechanism in case of execution errors.
        Sends the entire context (system, user, assistant messages) plus
        the error to the model on each retry so it can attempt to correct.
        """

        logging.info("🧑‍💻 Executing code")

        # We keep track of two different sets of messages
        # current_messages contains all correction attemps and the error messages to inform the model about already tried corrections
        # inner_current_messages contains only the messages for the most recent correction attempt so the parser can focus only on the current corrected code blob
        current_messages = messages.copy()
        inner_current_messages = messages.copy()
        execute_retries = retries

        while execute_retries > 0:
            execution_result, execution_logs, execution_error = self.python_interpreter(
                code
            )
            logging.debug(f"Execution result: {execution_result}")
            logging.debug(f"Execution error: {execution_error}")
            logging.debug(f"Python Interpreter state: {self.python_interpreter.state}")

            if execution_error:
                logging.error(f"❌ Failed to execute code: {execution_error}")
                execute_retries -= 1

                if execute_retries == 0:
                    logging.error(
                        f"❌ Failed to execute code after {retries} retries: {execution_error}"
                    )
                    return None

                # Prompt LLM to correct the code based on the execution error
                error_message = f"An error occurred during code execution: {execution_error}"
                logging.info("🔧 Attempting to correct code based on execution error.")

                current_messages.append(UserMessage(error_message))
                inner_current_messages.append(UserMessage(error_message))

                try:
                    corrected_code_blob = self.llm.generate(
                        current_messages, stop_sequences=["<end_code>"]
                    )
                    
                    current_messages.append(AssistantMessage(corrected_code_blob))
                    inner_current_messages.append(AssistantMessage(corrected_code_blob))

                    try:
                        # Parse the corrected code
                        code = self._parse_with_retries(
                            corrected_code_blob, inner_current_messages
                        )
                    except Exception as parse_e:
                        # Revert to the original messages so next iteration doesn't accumulate broken attempts
                        inner_current_messages = messages.copy()
                        continue

                except Exception as llm_e:
                    logging.error(f"❌ LLM failed to generate corrected code: {llm_e}")
                    return None

            else:
                # If successful execution but no final result
                if not execution_result:
                    message = "\n".join(execution_logs)
                    self.message_ledger.add(StepResultMessage(message))
                    logging.info("✔️ Successfully executed code with no final result.")
                    return None

                # Otherwise, return the final result
                return execution_result

        return None

    def _planning_step(self, task: str, is_first_step: bool, step: int):
        logging.debug(f"Planning step: is_first_step={is_first_step}, step={step}")

        if is_first_step:
            # Collect initial facts
            try:
                logging.info("🧠 Collecting facts")
                system_prompt_facts = SystemMessage(SYSTEM_PROMPT_FACTS)
                user_prompt_facts = UserMessage(USER_PROMPT_FACTS.format(task=task))
                facts = self.llm.generate([system_prompt_facts, user_prompt_facts])
                self.facts = facts
                logging.debug(f"Generated facts: {facts}")
            except Exception as e:
                logging.info(f"❌ Failed to collect facts {str(e)}")
                raise Exception(f"Error when generating model output:\n{str(e)}")

            # Generate plan
            try:
                logging.info("🧠 Generating plan")
                system_prompt_plan = SystemMessage(SYSTEM_PROMPT_PLAN)
                user_prompt_plan = UserMessage(
                    USER_PROMPT_PLAN.format(
                        task=task,
                        tool_descriptions=self.tool_descriptions,
                        facts=facts,
                    ),
                )
                plan = self.llm.generate(
                    [system_prompt_plan, user_prompt_plan],
                    stop_sequences=["<end_plan>"],
                )
                self.plan = plan
                logging.debug(f"Generated plan: {plan}")
            except Exception as e:
                logging.info(f"❌ Failed to generate plan. {str(e)}")
                raise Exception(f"Error when generating model output:\n{str(e)}")

            # Store plan in conversation
            try:
                self.message_ledger.add(PlanMessage(facts, plan))
                logging.debug("Appended initial plan and facts to message_ledger.")
            except Exception as e:
                logging.info(f"❌ Failed to generate plan. {str(e)}")
                raise Exception(
                    f"Error when generating final plan redaction:\n{str(e)}"
                )

        else:
            # Update facts
            try:
                logging.info("🧠 Updating facts")
                system_prompt_update_facts = SystemMessage(SYSTEM_PROMPT_FACTS_UPDATE)
                user_prompt_update_facts = UserMessage(
                    USER_PROMPT_FACTS_UPDATE.format(
                        task=task,
                        facts=self.facts,
                        history=self.message_ledger.summarize(),  # TODO summarize only the messages since the last step
                    ),
                )

                facts = self.llm.generate(
                    [system_prompt_update_facts, user_prompt_update_facts]
                )
                self.facts = facts
                logging.debug(f"Updated facts: {facts}")
            except Exception as e:
                logging.info(f"❌ Failed to collect facts. {str(e)}")
                raise Exception(f"Error when generating model output:\n{str(e)}")

            # Update plan
            try:
                logging.info("🧠 Updating plan")

                logging.debug("Tool Descriptions: ", self.tool_descriptions)
                logging.debug(
                    "Python Interpreter State: ", self.python_interpreter.state
                )

                system_prompt_update_plan = SystemMessage(SYSTEM_PROMPT_PLAN_UPDATE)
                user_prompt_update_plan = UserMessage(
                    USER_PROMPT_PLAN_UPDATE.format(
                        task=task,
                        tool_descriptions=self.tool_descriptions,
                        interpreter_state=json.dumps(self.python_interpreter.state),
                        facts=self.facts,
                        plan=self.plan,
                    ),
                )

                plan = self.llm.generate(
                    [system_prompt_update_plan, user_prompt_update_plan],
                    stop_sequences=["<end_plan>"],
                )
                self.plan = plan
                logging.debug(f"Updated plan: {plan}")
            except Exception as e:
                logging.info(f"❌ Failed to update plan. {str(e)}")
                raise Exception(f"Error when generating model output:\n{str(e)}")

            # Store updated plan in conversation
            try:
                self.message_ledger.add(PlanMessage(facts, plan))
                logging.debug("Added updated facts and plan to message ledger.")
            except Exception as e:
                logging.info(f"❌ Failed to update plan. {str(e)}")
                raise Exception(
                    f"Error when generating final plan redaction:\n{str(e)}"
                )

    def _generate_final_answer(self, task: str) -> str:
        logging.debug("Generating final answer based on interaction logs.")
        ledger_messages = self.message_ledger.copy()

        messages = [
            SystemMessage(
                "An agent tried to solve the user task but got stuck. Analyze the messages and provide a final response. This could mean, telling the user, that it was not possible to solve the given task."
            ),
            *ledger_messages,
            TaskMessage(f"Based on the above, please provide an answer to: {task}"),
        ]

        try:
            logging.debug("Generating final answer from model.")
            final_answer = self.llm.generate(messages)
            self.message_ledger.add(FinalAnswerMessage(final_answer))
            return final_answer
        except Exception as e:
            logging.debug("Error when generating final LLM output.")
            return f"Error when generating final LLM output:\n{str(e)}"

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

        except Exception as error:
            error_message = USER_PROMPT_PARSE_CODE_ERROR.format(
                code_blob=code_blob,
                error=error,
                pattern=pattern,
                tool_descriptions=self.tool_descriptions,
                interpreter_state=json.dumps(self.python_interpreter.state),
            )
            raise Exception(error_message)

    def _is_json_serializable(self, obj) -> bool:
        # Filter out modules
        if isinstance(obj, types.ModuleType):
            return False
        # Filter out classes
        if inspect.isclass(obj):
            return False

        # Attempt to JSON-serialize the value
        try:
            json.dumps(obj)
            return True
        except (TypeError, OverflowError):
            return False

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
