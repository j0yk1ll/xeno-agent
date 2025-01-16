from src.utils.messages import AssistantMessage, UserMessage


class ErrorMessage(UserMessage):
    def __init__(self, error: str):
        super().__init__(f"[ERROR] {error}\n")


class TaskMessage(UserMessage):
    def __init__(self, task: str):
        super().__init__(f"[TASK] {task}")


class PlanMessage(AssistantMessage):
    def __init__(self, facts: str, plan: str):
        super().__init__(
            f"[FACTS] {facts.strip()}\n[PLAN] {plan.strip()}",
        )


class ToolErrorMessage(UserMessage):
    def __init__(self, error: str):
        super().__init__(f"[TOOL ERROR] {error}\n")

class StepResultMessage(AssistantMessage):
    def __init__(self, result: str):
        super().__init__(f"[STEP RESULT] {result}")


class FinalAnswerMessage(AssistantMessage):
    def __init__(self, answer: str):
        super().__init__(f"[FINAL ANSWER] {answer}")
