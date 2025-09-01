import logging

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from pydantic import BaseModel

from state import ReflectionState


class GeneratorGraph(StateGraph):
    def create(self):
        def analyst_node(state: ReflectionState):
            response = self.context_schema.llm.invoke(
                self.context_schema.system_prompt
                + "\n\n"
                + "\n".join([m.content for m in state.messages])
            )
            return {"messages": [response], "reflection_count": state.reflection_count}

        self.add_node("generate", analyst_node)
        self.set_entry_point("generate")
        self.add_edge("generate", END)
        return self.compile()


class ReflectorGraph(StateGraph):
    def create(self):
        class AuditorOutput(BaseModel):
            is_credible: bool
            critique: str

        def auditor_node(state: ReflectionState):
            analysis_message = state.messages[-1]
            prompt_for_auditor = f"Please audit the following CVE analysis and respond in JSON format:\n\n{analysis_message.content}"
            critique_obj: AuditorOutput = self.context_schema.llm.invoke(
                self.context_schema.system_prompt + "\n\n" + prompt_for_auditor
            )

            if not critique_obj.is_credible:
                logging.warning(
                    f"Analysis failed audit. Critique: {critique_obj.critique}"
                )
                return {
                    "messages": [HumanMessage(content=critique_obj.critique)],
                    "reflection_count": state.reflection_count + 1,
                }
            else:
                logging.info("Analysis passed audit.")
                return {
                    "messages": [AIMessage(content="Audit passed.")],
                    "reflection_count": state.reflection_count + 1,
                }

        self.add_node("reflect", auditor_node)
        self.set_entry_point("reflect")
        self.add_edge("reflect", END)
        return self.compile()


class ReflectionGraph(StateGraph):
    def __init__(self, generator_graph, reflector_graph):
        super().__init__(ReflectionState)
        self.generator_graph = generator_graph
        self.reflector_graph = reflector_graph

    def should_reflect(self, state: ReflectionState):
        if state.reflection_count >= 2:
            logging.warning("Reflection limit reached, exiting loop.")
            return END
        last_msg = state.messages[-1]
        if isinstance(last_msg, AIMessage) and last_msg.content == "Audit passed":
            return END
        return "generator"

    def create(self):
        self.add_node("generator", self.generator_graph)
        self.add_node("reflector", self.reflector_graph)
        self.add_edge(START, "generator")
        self.add_edge("generator", "reflector")
        self.add_conditional_edges("reflector", self.should_reflect)
        return self.compile()
