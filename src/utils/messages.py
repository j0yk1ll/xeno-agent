from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(dict):
    role: str
    content: str

    def __init__(self, role: MessageRole, content: str):
        dict.__init__(self, role=role.value, content=content)

        self.role = role.value
        self.content = content


class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__(MessageRole.SYSTEM, content)


class UserMessage(Message):
    def __init__(self, content: str):
        super().__init__(MessageRole.USER, content)


class AssistantMessage(Message):
    def __init__(self, content: str):
        super().__init__(MessageRole.ASSISTANT, content)
