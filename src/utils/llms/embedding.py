import logging
from typing import List, Optional

import litellm
import numpy as np
from src.utils.llms.base import BaseLLM

# Disable verbose logging from litellm
litellm.set_verbose = False

class EmbeddingLLM(BaseLLM):
    """
    Handles embedding requests.
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

    def call(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a given text.

        Args:
            text (str): The text to embed.

        Returns:
            np.ndarray: The embedding vector as a NumPy array.
        """
        try:
            # Acquire rate limit before making the API call, if enabled
            self._acquire_rate_limit()

            embedding_response = litellm.embedding(
                model=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key,
                input=text,
            )
            embedding = embedding_response["data"][0]["embedding"]
            logging.debug(
                f"Generated embedding for text: {text[:30]}..."
            )  # Log first 30 chars
            return np.array(embedding, dtype=np.float32)

        except litellm.APIConnectionError as e:
            # Quit the application if a connection error occurs
            logging.error(f"A litellm.APIConnectionError occurred: {e}")
            raise SystemExit(
                "Encountered an API connection error. Exiting the agent now."
            ) from e

        except Exception as e:
            logging.error(f"Error embedding text: {e}")
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
            self._acquire_rate_limit()

            embedding_response = litellm.embedding(
                model=self.model_id,
                api_base=self.api_base,
                api_key=self.api_key,
                input=batch,
            )
            data = embedding_response["data"]
            embeddings = [entry["embedding"] for entry in data]
            logging.debug(f"Generated embeddings for batch of size: {len(batch)}")
            return np.array(embeddings, dtype=np.float32)

        except litellm.APIConnectionError as e:
            # Quit the application if a connection error occurs
            logging.error(f"A litellm.APIConnectionError occurred: {e}")
            raise SystemExit(
                "Encountered an API connection error. Exiting the agent now."
            ) from e

        except Exception as e:
            logging.error(f"Error embedding batch: {e}")
            raise
