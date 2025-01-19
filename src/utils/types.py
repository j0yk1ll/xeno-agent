from enum import Enum


class FileType(str, Enum):
    IMAGE = "image"  # PNG
    AUDIO = "audio"  # WAV
