import logging
from typing import List, Optional

import litellm

from src.utils.messages import Message
from src.utils.llms.base import BaseLLM
from src.utils.llms.utils import remove_stop_sequences

# Disable verbose logging from litellm
litellm.set_verbose = False

class CompletionLLM(BaseLLM):
    """
    Handles text completion requests.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
    ):
        super().__init__(
            model_id=model_id,
            api_base=api_base,
            api_key=api_key,
            requests_per_minute=requests_per_minute,
        )

    def call(
        self,
        messages: List[Message],
        stop_sequences: Optional[List[str]] = None,
        max_tokens: int = 1500,
    ) -> str:
        """
        Generate text given a list of messages.

        Args:
            messages (List[Message]): A list of Message objects containing roles and content.
            stop_sequences (Optional[List[str]]): A list of stop sequences to terminate generation.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated text content.
        """
        if not isinstance(messages, list):
            logging.warning(
                "Messages should be a list of Message objects: %s", messages
            )
            raise TypeError("Messages should be a list of Message objects")

        if stop_sequences is None:
            stop_sequences = []

        # Convert custom Message objects to litellm's expected format
        litellm_messages = [{"role": m.role, "content": m.content} for m in messages]

        logging.debug(f"CompletionLLM Input Messages: {litellm_messages}")

        try:
            # Acquire rate limit before making the API call, if enabled
            self._acquire_rate_limit()

            response = litellm.completion(
                model=self.model_id,
                messages=litellm_messages,
                stop=stop_sequences,
                max_tokens=max_tokens,
                api_base=self.api_base,
                api_key=self.api_key,
            )

            # Extract content from response
            content = response.choices[0].message.content
            content = remove_stop_sequences(content, stop_sequences)

            logging.debug(f"CompletionLLM Output Content: {content}")

            return content

        except litellm.APIConnectionError as e:
            # Quit the application if a connection error occurs
            logging.warning(f"A litellm.APIConnectionError occurred: {e}")
            raise SystemExit(
                "Encountered an API connection error. Exiting the agent now."
            ) from e

        except Exception as e:
            logging.warning(f"An error occurred generating model output: {e}")
            raise
