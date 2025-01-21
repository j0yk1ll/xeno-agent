import base64
import io
import logging
from typing import List, Optional

import PIL
import litellm
from src.utils.llms.base import BaseLLM
from src.utils.llms.utils import remove_stop_sequences

# Disable verbose logging from litellm
litellm.set_verbose = False

class VisionLLM(BaseLLM):
    """
    Handles vision (image processing) requests.
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
        image: PIL.Image.Image,
        instruction: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        max_tokens: int = 1500,
    ) -> str:
        """
        Generate text given an image.

        Args:
            image (PIL.Image.Image): An image.
            instruction (Optional[str]): An instruction.
            stop_sequences (Optional[List[str]]): A list of stop sequences to terminate generation.
            max_tokens (int): The maximum number of tokens to generate.

        Returns:
            str: The generated text content.
        """

        if stop_sequences is None:
            stop_sequences = []

        if not instruction:
            instruction = "What do you see? Give a compressed description."

        # Convert the PIL.Image to a base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        data = base64.b64encode(img_bytes).decode()

        # Prepare messages in litellm's expected format
        litellm_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{data}"
                        },
                    },
                ],
            }
        ]

        logging.debug(f"VisionLLM Input Messages: {litellm_messages}")

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

            logging.debug(f"VisionLLM Output Content: {content}")

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
