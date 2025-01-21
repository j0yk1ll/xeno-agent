from typing import List


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
            content = content[: -len(stop_seq)]
    return content