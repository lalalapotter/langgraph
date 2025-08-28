import logging
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel


class MessagesState(TypedDict):
    messages: List[BaseMessage]

class ReflectionState(MessagesState):
    reflection_count: int

class GeneratorGraph(StateGraph):
    def __init__(self, llm, system_message):
        super().__init__(ReflectionState)
        self.llm = llm
        self.system_message = system_message
    
    def create(self):
        def analyst_node(state: ReflectionState):
            response = self.llm.invoke(self.system_message + "\n\n" + "\n".join([m.content for m in state['messages']]))
            return {"messages": [response], "reflection_count": state.get("reflection_count", 0)}
        graph = StateGraph(ReflectionState)
        graph.add_node("generate", analyst_node)
        graph.set_entry_point("generate")
        graph.add_edge("generate", END)
        return graph.compile()

class ReflectorGraph(StateGraph):
    def __init__(self, llm, system_message):
        super().__init__(ReflectionState)
        self.llm = llm
        self.system_message = system_message

    def create(self):
        class AuditorOutput(BaseModel):
            is_credible: bool
            critique: str
        def auditor_node(state: ReflectionState):
            analysis_message = state['messages'][-1]
            prompt_for_auditor = f"Please audit the following CVE analysis and respond in JSON format:\n\n{analysis_message.content}"
            critique_obj: AuditorOutput =self.llm.invoke(self.system_message + "\n\n" + prompt_for_auditor)
            
            if not critique_obj.is_credible:
                logging.warning(f"Analysis failed audit. Critique: {critique_obj.critique}")
                return {"messages": [HumanMessage(content=critique_obj.critique)], "reflection_count": state.get("reflection_count", 0)}
            else:
                logging.info("Analysis passed audit.")
                return {"messages": [AIMessage(content="Audit passed.")], "reflection_count": state.get("reflection_count", 0)}
        graph = StateGraph(ReflectionState)
        graph.add_node("reflect", auditor_node)
        graph.set_entry_point("reflect")
        graph.add_edge("reflect", END)
        return graph.compile()
        

class ReflectionGraph(StateGraph):
    def __init__(self, generator_graph, reflector_graph):
        super().__init__(ReflectionState)
        self.generator_graph = generator_graph
        self.reflector_graph = reflector_graph

    def should_reflect(self, state: ReflectionState):
        if state["reflection_count"] >= 3:
            logging.warning("Reflection limit reached, exiting loop.")
            return END
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.content == "Audit passed":
            return END
        state["reflection_count"] += 1
        return "generator"

    def create(self):
        rgraph = StateGraph(ReflectionState)
        rgraph.add_node("generator", self.generator_graph)
        rgraph.add_node("reflector", self.reflector_graph)
        rgraph.add_edge(START, "generator")
        rgraph.add_edge("generator", "reflector")
        rgraph.add_conditional_edges("reflector", self.should_reflect)
        return rgraph.compile()

