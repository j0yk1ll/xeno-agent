import logging
from src.utils.tool import Tool


class DoNothingTool(Tool):
    name = "do_nothing"
    description = "Do nothing."
    inputs = {}
    output_type = "any"

    def forward(self) -> None:
        logging.info(f"🧰 Using tool: {self.name}")