from __future__ import annotations
from typing import List, Literal, cast, Any, Callable

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from state import MessageState
from langchain_tavily import TavilySearch


class ReActGraph(StateGraph):
    def create(self):
        def search(query: str):
            """Search for general web results.

            This function performs a search using the Tavily search engine, which is designed
            to provide comprehensive, accurate, and trusted results. It's particularly useful
            for answering questions about current events.
            """
            wrapped = TavilySearch(max_results=1)
            return cast(dict[str, Any], wrapped.ainvoke({"query": query}))

        TOOLS: List[Callable[..., Any]] = [search]

        def call_model(state: MessageState):
            llm = self.context_schema.llm
            system_prompt = self.context_schema.system_prompt
            response = cast(
                AIMessage,
                llm.invoke(
                    [{"role": "system", "content": system_prompt}, *state.messages]
                ),
            )
            state.messages = [response]
            return state

        def route_model_output(state: MessageState) -> Literal["__end__", "tools"]:
            last_message = state.messages[-1]
            if not isinstance(last_message, AIMessage):
                raise ValueError(
                    f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
                )
            if not last_message.tool_calls:
                return "__end__"
            return "tools"

        self.add_node(call_model)
        self.add_node("tools", ToolNode(TOOLS))
        self.add_edge("__start__", "call_model")

        self.add_conditional_edges(
            "call_model",
            route_model_output,
        )

        self.add_edge("tools", "call_model")
        return self.compile(name="ReAct Agent")
