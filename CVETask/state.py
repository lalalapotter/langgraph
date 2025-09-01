from __future__ import annotations
from datetime import UTC, datetime
from dataclasses import dataclass, field, fields
from typing import Dict, List, Literal, cast, Annotated, Any, Callable, Sequence
from typing_extensions import Annotated

from dataclasses import dataclass, field
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

from langgraph.managed import IsLastStep


@dataclass
class MessageState:
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )

@dataclass
class ReflectionState(MessageState):
    reflection_count: int = 0
