from typing import Dict, Optional
import logging

import PIL

from src.utils.tool import Tool
from src.utils.memory_manager import MemoryManager

class SaveMemoryTool(Tool):
    name = "save_memory"
    description = (
        "Store a memory."
    )
    inputs = {
        "text": {
            "type": "string",
            "description": "Some short descriptive text."
        },
        "file_id": {
            "type": "string",
            "description": "An optional unique id of the image to attach to the memory.",
            "optional": True
        }
    }
    output_type = "null"

    def __init__(self, memory_manager: MemoryManager, images: Dict[str, PIL.Image]):
        """
        :param memory_manager: An instance of the memory manager that provides the insert_image method.
        :param images: An instance of the memory manager that provides the insert_image method.
        """
        self.memory_manager = memory_manager
        self.images = images
        super().__init__()

    def forward(self, text: str, file_id: Optional[str] = None) -> None:
        """
        :param text: The text to be saved.
        :param file: The unique file id of the file to be saved.
        """

        if file_id:
            file = self.images[file_id]
            self.memory_manager.insert_image(text, file)
        else:
            self.memory_manager.insert_text(text)
        logging.debug(f"Inserted memory.")
