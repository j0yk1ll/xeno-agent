import logging
from typing import Callable
from src.utils.tool import Tool


class TalkTool(Tool):
    name = "talk"
    description = "Talk to the user."
    inputs = {
        "utterance": {"type": "string", "description": "The words you want to say."}
    }

    def __init__(self, on_observation: Callable[[str], None]):
        self.on_observation = on_observation
        super().__init__()

    def forward(self, utterance: str):
        logging.info(f"🧰 Using tool: {self.name}")
        self.on_observation(utterance)