import logging
import re
from typing import Callable, Dict, List, Optional
import uuid

from src.proxy_agent.tools.click_element import ClickElementTool
from src.proxy_agent.tools.close_browser_context import CloseBrowserContextTool
from src.proxy_agent.tools.create_browser_context import CreateBrowserContextTool
from src.proxy_agent.tools.run_code import RunCodeTool
from src.proxy_agent.tools.extract_page_content import ExtractPageContentTool
from src.proxy_agent.tools.get_dropdown_options import GetDropdownOptionsTool
from src.proxy_agent.tools.go_back import GoBackTool
from src.proxy_agent.tools.go_to_url import GoToUrlTool
from src.proxy_agent.tools.input_text import InputTextTool
from src.proxy_agent.tools.open_tab import OpenTabTool
from src.proxy_agent.tools.scroll_page import ScrollPageTool
from src.proxy_agent.tools.scroll_to_text import ScrollToTextTool
from src.proxy_agent.tools.search_google import SearchGoogleTool
from src.proxy_agent.tools.select_dropdown_option import SelectDropdownOptionTool
from src.proxy_agent.tools.send_keys import SendKeysTool
from src.proxy_agent.tools.switch_tab import SwitchTabTool
from src.proxy_agent.tools.use_terminal import UseTerminalTool
from src.proxy_agent.tools.do_nothing import DoNothingTool
from src.proxy_agent.tools.talk import TalkTool
from src.proxy_agent.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    USER_PROMPT_PARSE_CODE_ERROR,
)
from src.utils.local_python_interpreter import LocalPythonInterpreter
from src.utils.tool import Tool
from src.utils.messages import AssistantMessage, Message, SystemMessage, UserMessage
from src.utils.types import FileType
from src.utils.browser.browser import Browser
from src.utils.llms.completion import CompletionLLM


class ProxyAgent:
    """
    Agent class that solves tasks step by step using a ReAct-like framework.
    It performs cycles of action (LLM-generated code) and observation (execution results).
    """

    def __init__(
        self,
        browser: Browser,
        completion_llm: CompletionLLM,
        on_response: Callable
    ):
        logging.debug("Initializing Proxy Agent.")
        self.agent_name = self.__class__.__name__

        self.browser = browser

        self.completion_llm = completion_llm

        self.on_response = on_response

        self.tools: List[Tool] = [
            TalkTool(self._talk),
            DoNothingTool(),
            RunCodeTool(on_observation=self._add_observation),
            UseTerminalTool(on_observation=self._add_observation),
            CreateBrowserContextTool(browser=self.browser, on_observation=self._add_observation),
            CloseBrowserContextTool(on_observation=self._add_observation),
            ClickElementTool(on_observation=self._add_observation),
            ExtractPageContentTool(on_observation=self._add_observation),
            GetDropdownOptionsTool(on_observation=self._add_observation),
            GoBackTool(on_observation=self._add_observation),
            GoToUrlTool(on_observation=self._add_observation),
            InputTextTool(on_observation=self._add_observation),
            OpenTabTool(on_observation=self._add_observation),
            ScrollPageTool(on_observation=self._add_observation),
            ScrollToTextTool(on_observation=self._add_observation),
            SearchGoogleTool(on_observation=self._add_observation),
            SelectDropdownOptionTool(on_observation=self._add_observation),
            SendKeysTool(on_observation=self._add_observation),
            SwitchTabTool(on_observation=self._add_observation),
        ]

        self.python_interpreter = LocalPythonInterpreter(self.tools)

        self.tool_descriptions = "\n".join(
            [
                f"-{tool.name}({tool.inputs}): {tool.description}"
                for tool in self.tools
            ]
        )

        self.system_prompt = SYSTEM_PROMPT.format(
            tool_descriptions=self.tool_descriptions
        )

        self.observations = []
        self.unprocessed_observations = []
        self.observation_images = {}

    def update_completion_llm(self, completion_llm: CompletionLLM):
        self.completion_llm = completion_llm

    def add_observation(self, source: str, text: str, files: List[Dict[str, any]]):

        observations = []

        for file in files:
            file_type = file['type']
            file_object = file['object']

            file_id = uuid.uuid4()

            if file_type == FileType.IMAGE:
                self.observation_images[file_id] = file_object
                observations.append(f"You received an image from {source}. It's unique id is {file_id}.")

            elif file_type == FileType.AUDIO:
                # ignore for now. Might be important later for dynamic speaker identification or meetings.
                continue
            
        observations.append(text)

        self._add_observations(observations)

    def _add_observations(self, observations: List[str]):
        self.observations += observations
        self.unprocessed_observations += observations

        self._process_observations()

    def _add_observation(self, observation: str):
        self.observations.append(observation)
        self.unprocessed_observations.append(observation)

        self._process_observations()

    def _process_observations(self):
        try:
            logging.info("🏃 Generating action")

            most_recent_observation = self.observations[-1]
            context_observations = self.observations[-21:-1] # Get the previous 20 observations for context

            user_prompt = USER_PROMPT.format(
                observation=most_recent_observation,
                context=context_observations,
            )

            messages = [
                SystemMessage(self.system_prompt),
                UserMessage(user_prompt),
            ]

            code_blob = self.completion_llm.call(messages)
        except Exception as e:
            logging.error(f"❌ Failed to generate action. {str(e)}.")
            raise Exception(f"Error when generating model output:\n{str(e)}")

        messages_with_code_blob = [
            *messages,
            AssistantMessage(code_blob),
        ]  # Add generated code to messages
        # Parse the generated code with retries
        try:
            code = self._parse_with_retries(code_blob, messages_with_code_blob)

            messages_with_parsed_code = [*messages, AssistantMessage(code)]
            # Execute the parsed code with retries
            self._execute_with_retries(code, messages_with_parsed_code)
        except Exception as e:
            # If code couldn't be corrected or executed within the given retries let the agent know and exit
            self.add_observation(str(e))
            return

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

    def _execute_with_retries(self, code: str, messages: List[Message]):
        logging.info("🧑‍💻 Executing code")

        # We keep track of two different sets of messages
        # current_messages contains all correction attemps and the error messages to inform the model about already tried corrections
        # inner_current_messages contains only the messages for the most recent correction attempt so the parser can focus only on the current corrected code blob        
        current_messages = messages.copy()
        inner_current_messages = messages.copy()
        execute_retries = 5
        while execute_retries > 0:
            _, _, error = self.python_interpreter(code)

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
                    corrected_code_blob = self.completion_llm.call(
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

    def _talk(self, utterance: str):
        logging.debug(f"_on_talk with {utterance}")
        self.on_response(utterance)