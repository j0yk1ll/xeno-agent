import logging
from typing import Callable
from src.utils.tool import Tool


class TalkTool(Tool):
    name = "talk"
    description = "Talk to the user."
    inputs = {
        "utterance": {"type": "string", "description": "The words you want to say."}
    }
    output_type = "any"

    def __init__(self, on_result: Callable):
        self.on_result = on_result
        super().__init__()

    def forward(self, utterance: str) -> str:
        logging.info(f"🧰 Using tool: {self.name}")
        self.on_result(utterance)