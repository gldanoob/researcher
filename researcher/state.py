import operator
from typing import Annotated

from langchain.agents import AgentState
from langgraph.graph import MessagesState


class ResearchState(AgentState):
    ideas: Annotated[list[str], operator.add]
    proposal: str
    bibliography: str
    literature_review: str
    final_paper: str
