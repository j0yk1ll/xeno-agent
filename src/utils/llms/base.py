import litellm
from typing import Optional

from src.utils.llms.rate_limiter import RateLimiter

class BaseLLM:
    """
    A base class providing shared functionalities for Completion, Embedding, and Vision LLMs.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        requests_per_minute: Optional[int] = None,
    ):
        if not model_id:
            raise ValueError("A model_id must be provided.")

        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(requests_per_minute)

    def _acquire_rate_limit(self):
        """
        Acquires a rate limit slot if rate limiting is enabled.
        """
        self.rate_limiter.acquire_slot()
