from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Wrapper around a generative model for bias evaluation."""

    @abstractmethod
    def generate(self, prompts: list[str], max_new_tokens: int = 40) -> list[str]:
        """Generate continuations for each prompt. Returns completions only (no prompt echo)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
