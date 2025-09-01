import os
from dataclasses import dataclass, field, fields
from typing_extensions import Annotated
from typing import Annotated

SYSTEM_PROMPT = """You are a helpful AI assistant."""


@dataclass(kw_only=True)
class Context:
    def __init__(self,
                 llm,
                 system_prompt: str = SYSTEM_PROMPT,
                 ):
        self.llm = llm
        self.system_prompt = system_prompt

    def __post_init__(self) -> None:
        """Fetch env vars for attributes that were not passed as args."""
        for f in fields(self):
            if not f.init:
                continue

            if getattr(self, f.name) == f.default:
                setattr(self, f.name, os.environ.get(f.name.upper(), f.default))