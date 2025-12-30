from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import InjectedState
from langgraph.types import Command, Send
from typing_extensions import Literal


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        state: Annotated[MessagesState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        # highlight-next-line
        return Command(
            # highlight-next-line
            goto=agent_name,  # (1)!
            # highlight-next-line
            update={**state, "messages": state["messages"] + [tool_message]},  # (2)!
            # highlight-next-line
            graph=Command.PARENT,  # (3)!
        )

    return handoff_tool


@tool("hand_off_to_agents", description="Hand off the research task to multiple agents in parallel.")
def handoff_to_multiple_agents_tool(
    agent_names: Annotated[
        list[Literal["literature_review_agent_1", "literature_review_agent_2", "proposal_writer_agent"]],
        "List of names of agents to hand off the research task to.",
    ],
    task_descriptions: Annotated[
        list[str],
        "Description of what the next agents should do, including all of the relevant context and key information. Should be in the same order as agent_names.",
    ],
    state: Annotated[MessagesState, InjectedState],
) -> Command:
    commands = []
    for agent_name, task_description in zip(agent_names, task_descriptions):
        task_description_message = HumanMessage(content=task_description)
        agent_input = {**state, "messages": [task_description_message]}
        commands.append(Send(agent_name, agent_input))

    return Command(
        goto=commands,
        graph=Command.PARENT,
    )


def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            # highlight-next-line
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool
