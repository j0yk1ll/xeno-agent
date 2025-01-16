# llm.py

import logging
import time
import numpy as np
import litellm
from typing import List, Optional, Tuple
from threading import Lock

from src.utils.messages import Message  # Ensure this import is correct based on your project structure

# Disable verbose logging from litellm
litellm.set_verbose = False


def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    """
    Removes any stop sequences from the end of the content.

    Args:
        content (str): The content string to process.
        stop_sequences (List[str]): A list of stop sequences to remove.

    Returns:
        str: The content string with stop sequences removed from the end.
    """
    for stop_seq in stop_sequences:
        if content and content.endswith(stop_seq):
            content = content[:-len(stop_seq)]
    return content


class LLM:
    def __init__(
        self,
        completion_model_id: str,
        embedding_model_id: str,
        completion_api_base: Optional[str] = None,
        completion_api_key: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        completion_requests_per_minute: Optional[int] = None,  # Optional rate limit for completions
        embedding_requests_per_minute: Optional[int] = None,   # Optional rate limit for embeddings
    ):
        """
        Initializes the LLM.

        Args:
            completion_model_id (str): The identifier for the LLM model used for completions.
            embedding_model_id (str): The identifier for the LLM model used for embeddings.
            completion_api_base (str, optional): The base URL for the completion API.
            completion_api_key (str, optional): The API key for authentication (completions).
            embedding_api_base (str, optional): The base URL for the embedding API.
            embedding_api_key (str, optional): The API key for embedding model.
            completion_requests_per_minute (int, optional): Max completion requests per minute.
            embedding_requests_per_minute (int, optional): Max embedding requests per minute.
        """
        if not completion_model_id:
            raise ValueError("A completion_model_id must be provided.")
        if not embedding_model_id:
            raise ValueError("An embedding_model_id must be provided.")

        self.completion_model_id = completion_model_id
        self.completion_api_base = completion_api_base
        self.completion_api_key = completion_api_key

        self.embedding_model_id = embedding_model_id
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key

        # Rate limiting parameters
        self.completion_requests_per_minute = completion_requests_per_minute
        self.embedding_requests_per_minute = embedding_requests_per_minute

        # Initialize rate limiting only if limits are provided
        if self.completion_requests_per_minute is not None:
            self._completion_lock = Lock()
            self._completion_window_start = time.time()
            self._completion_request_count = 0
        else:
            self._completion_lock = None  # Indicate that rate limiting is disabled for completions

        if self.embedding_requests_per_minute is not None:
            self._embedding_lock = Lock()
            self._embedding_window_start = time.time()
            self._embedding_request_count = 0
        else:
            self._embedding_lock = None  # Indicate that rate limiting is disabled for embeddings

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,  # Default level; can be adjusted as needed
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _check_rate_limit(
        self,
        lock: Lock,
        window_start: float,
        request_count: int,
        max_requests: int
    ) -> Tuple[float, int]:
        """
        Checks and updates the rate limit counters.

        Args:
            lock (Lock): A threading lock to ensure thread safety.
            window_start (float): The start time of the current window.
            request_count (int): The number of requests made in the current window.
            max_requests (int): The maximum number of allowed requests per window.

        Returns:
            Tuple[float, int]: Updated window_start and request_count.
        """
        with lock:
            current_time = time.time()
            elapsed = current_time - window_start

            if elapsed >= 60:
                # Reset the window
                window_start = current_time
                request_count = 1
                self.logger.debug("Rate limit window reset.")
            else:
                if request_count < max_requests:
                    request_count += 1
                    self.logger.debug(f"Incremented request count: {request_count}/{max_requests}")
                else:
                    # Wait until the window resets
                    sleep_time = 60 - elapsed
                    self.logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                    time.sleep(sleep_time)
                    window_start = time.time()
                    request_count = 1
                    self.logger.debug("Rate limit window reset after sleep.")

            return window_start, request_count

    def _acquire_completion_rate_limit(self):
        """
        Acquires the rate limit slot for a completion request.
        """
        if self.completion_requests_per_minute is not None and self._completion_lock is not None:
            self._completion_window_start, self._completion_request_count = self._check_rate_limit(
                self._completion_lock,
                self._completion_window_start,
                self._completion_request_count,
                self.completion_requests_per_minute
            )

    def _acquire_embedding_rate_limit(self):
        """
        Acquires the rate limit slot for an embedding request.
        """
        if self.embedding_requests_per_minute is not None and self._embedding_lock is not None:
            self._embedding_window_start, self._embedding_request_count = self._check_rate_limit(
                self._embedding_lock,
                self._embedding_window_start,
                self._embedding_request_count,
                self.embedding_requests_per_minute
            )

    def generate(
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
            self.logger.warning("Messages should be a list of Message objects: %s", messages)
            raise TypeError("Messages should be a list of Message objects")

        if stop_sequences is None:
            stop_sequences = []

        # Convert custom Message objects to litellm's expected format
        litellm_messages = [{"role": m.role, "content": m.content} for m in messages]

        self.logger.debug(f"LLM Input Messages: {litellm_messages}")

        try:
            # Acquire rate limit before making the API call, if enabled
            self._acquire_completion_rate_limit()

            response = litellm.completion(
                model=self.completion_model_id,
                messages=litellm_messages,
                stop=stop_sequences,
                max_tokens=max_tokens,
                api_base=self.completion_api_base,
                api_key=self.completion_api_key,
            )

            # Extract content from response
            content = response.choices[0].message.content
            content = remove_stop_sequences(content, stop_sequences)

            self.logger.debug(f"LLM Output Content: {content}")

            return content

        except litellm.APIConnectionError as e:
            # Quit the application if a connection error occurs
            self.logger.warning(f"A litellm.APIConnectionError occurred: {e}")
            raise SystemExit("Encountered an API connection error. Exiting the agent now.") from e

        except Exception as e:
            self.logger.warning(f"An error occurred generating model output: {e}")
            raise

    def embed(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a given text.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The embedding vector as a NumPy array.
        """
        try:
            # Acquire rate limit before making the API call, if enabled
            self._acquire_embedding_rate_limit()

            embedding_response = litellm.embedding(
                model=self.embedding_model_id,
                api_base=self.embedding_api_base,
                api_key=self.embedding_api_key,
                input=text,
            )
            embedding = embedding_response["data"][0]["embedding"]
            self.logger.debug(f"Generated embedding for text: {text[:30]}...")  # Log first 30 chars
            return np.array(embedding, dtype=np.float32)

        except litellm.APIConnectionError as e:
            # Quit the application if a connection error occurs
            self.logger.error(f"A litellm.APIConnectionError occurred: {e}")
            raise SystemExit("Encountered an API connection error. Exiting the agent now.") from e

        except Exception as e:
            self.logger.error(f"Error embedding text: {e}")
            raise

    def embed_batch(self, batch: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            batch (List[str]): A list of texts to embed.

        Returns:
            np.ndarray: A 2D NumPy array where each row is an embedding vector.
        """
        try:
            # Acquire rate limit before making the API call, if enabled
            self._acquire_embedding_rate_limit()

            embedding_response = litellm.embedding(
                model=self.embedding_model_id,
                api_base=self.embedding_api_base,
                api_key=self.embedding_api_key,
                input=batch,
            )
            data = embedding_response["data"]
            embeddings = [entry["embedding"] for entry in data]
            self.logger.debug(f"Generated embeddings for batch of size: {len(batch)}")
            return np.array(embeddings, dtype=np.float32)

        except litellm.APIConnectionError as e:
            # Quit the application if a connection error occurs
            self.logger.error(f"A litellm.APIConnectionError occurred: {e}")
            raise SystemExit("Encountered an API connection error. Exiting the agent now.") from e

        except Exception as e:
            self.logger.error(f"Error embedding batch: {e}")
            raise