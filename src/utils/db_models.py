from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
from pony.orm import Database, Required, Optional, PrimaryKey, Set

db = Database()


class FileType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class File(db.Entity):
    id = PrimaryKey(UUID, default=uuid4)
    type = Required(FileType)
    path = Required(str)

    def before_insert(self):
        if self.type not in FileType:
            raise ValueError(
                f"Invalid file type: {self.type}. Allowed types are {[ft.value for ft in FileType]}."
            )


class Observation(db.Entity):
    id = PrimaryKey(UUID, default=uuid4)
    timestamp = Required(datetime, default=lambda: datetime.now(datetime.timezone.utc))
    content = Required(str)
    file = Optional(File)


entities = [File, Observation]
