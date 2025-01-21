import logging
from src.utils.tool import Tool


class DoNothingTool(Tool):
    name = "do_nothing"
    description = "Do nothing."
    inputs = {}

    def forward(self):
        logging.info(f"🧰 Using tool: {self.name}")