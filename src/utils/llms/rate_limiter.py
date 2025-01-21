import logging
from threading import Lock
import time
from typing import Optional


class RateLimiter:
    """
    A utility class to handle rate limiting for API requests.
    """

    def __init__(self, requests_per_minute: Optional[int]):
        if requests_per_minute is not None:
            self.lock = Lock()
            self.window_start = time.time()
            self.request_count = 0
            self.max_requests = requests_per_minute
        else:
            self.lock = None  # Rate limiting disabled

    def acquire_slot(self):
        """
        Acquires a rate limit slot, blocking if necessary until a slot is available.
        """
        if self.lock is None:
            return  # Rate limiting is disabled

        with self.lock:
            current_time = time.time()
            elapsed = current_time - self.window_start

            if elapsed >= 60:
                # Reset the window
                self.window_start = current_time
                self.request_count = 1
                logging.debug("Rate limit window reset.")
            else:
                if self.request_count < self.max_requests:
                    self.request_count += 1
                    logging.debug(
                        f"Incremented request count: {self.request_count}/{self.max_requests}"
                    )
                else:
                    # Wait until the window resets
                    sleep_time = 60 - elapsed
                    logging.info(
                        f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds."
                    )
                    time.sleep(sleep_time)
                    self.window_start = time.time()
                    self.request_count = 1
                    logging.debug("Rate limit window reset after sleep.")