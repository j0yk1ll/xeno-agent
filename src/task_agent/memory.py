import logging
import numpy as np
from typing import List

from src.utils.llm import LLM


class Memory(dict):
    text: str
    embedding: np.ndarray

    def __init__(self, text: str, embedding: np.ndarray):
        dict.__init__(self, text=text, embedding=embedding)

        self.text = text
        self.embedding = embedding


class EpisodicMemory:
    def __init__(self, llm: LLM):
        """
        Initialize the Memory object.

        Args:
            llm: A language model interface with an async callable interface.
        """
        self.llm = llm
        self.memories = []

    async def memorize(self, text: str) -> None:
        """
        Method for adding a memory.
        """
        try:
            logging.info("💾 Adding memory.")

            # Also embed the summary text
            embedding = await self.llm.embed(text)

            self.memories.append(
                Memory(text=text, embedding=self._normalize(embedding))
            )

        except Exception as e:
            logging.error(f"❌ Failed to add memory: {str(e)}", exc_info=True)

    async def get_memories(
        self, query: str, n_results: int = 3, distance_threshold: float = 0.8
    ) -> List[str]:
        """
        Retrieve the top n most relevant memories to the query, using cosine similarity.
        """
        try:
            query_vector = await self.llm.embed(query)

            # If we have no stored memories yet, return empty
            if not self.memories:
                return []

            query_vector_norm = self._normalize(query_vector)
            memory_vectors_norm = np.array(
                [memory.embedding for memory in self.memories]
            )

            # Compute cosine similarity
            cosine_similarities = np.dot(
                memory_vectors_norm, query_vector_norm.T
            )  # shape: (num_memories,)

            # Find the indices of the top n most similar
            top_n_indices = np.argsort(cosine_similarities)[-n_results:][::-1]

            # Filter out those below threshold
            top_memories = []
            for idx in top_n_indices:
                if cosine_similarities[idx] >= distance_threshold:
                    top_memories.append(self.memories[idx])

            logging.debug(f"Top {n_results} most similar memories: {top_memories}")
            return top_memories

        except Exception as e:
            logging.error(f"Error retrieving memories: {str(e)}")
            return []

    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v if norm == 0 else v / norm
