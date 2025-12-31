import asyncio
import os
import shutil

import aiosqlite
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (ModelCallLimitMiddleware,
                                         SummarizationMiddleware)
from langchain.messages import RemoveMessage
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import BraveSearch
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.messages import AIMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from researcher.state import ResearchState
from researcher.utils import tools
from researcher.utils.handoff import handoff_to_multiple_agents_tool
from researcher.utils.pretty_print import pretty_print_messages

load_dotenv()


def is_alive_patch(self):
    return True


if not hasattr(aiosqlite.Connection, "is_alive"):
    setattr(aiosqlite.Connection, "is_alive", is_alive_patch)


def read_prompt(agent_name: str) -> str:
    prompt_path = os.path.join("prompts", f"{agent_name}.txt")
    if not os.path.exists(prompt_path):
        raise ValueError(f"Prompt file for agent {agent_name} does not exist.")
    with open(prompt_path, "r") as f:
        return f.read()


# Turn all human messages into system messages before returning to supervisor

def convert_human_to_ai_messages(state: ResearchState) -> ResearchState:
    print("Changing human messages to AI messages (Except first message) before returning to supervisor...")

    converted_messages = []

    for msg in state["messages"][1:]:
        if msg.type == "human":
            converted_messages.append(AIMessage(content=msg.content, id=msg.id))

    return {
        "messages": converted_messages,
    }


async def main():
    async with aiosqlite.connect("researcher_state.db") as conn:
        model = ChatDeepSeek(model="deepseek-reasoner")
        checkpointer = AsyncSqliteSaver(conn)

        mcp_tools = await tools.get_arxiv_tools()

        sub_agents_tools = {
            "literature_reviewer_1": mcp_tools + tools.file_tools + [tools.brave_search_tool] + tools.request_tools,
            "literature_reviewer_2": mcp_tools + tools.file_tools + [tools.brave_search_tool] + tools.request_tools,
            "technical_writer": tools.file_tools + tools.request_tools + [tools.brave_search_tool],
            "peer_reviewer": tools.file_tools + tools.request_tools + [tools.brave_search_tool],
        }

        sub_agents = {
            name: create_agent(
                name=name,
                model=model,
                system_prompt=read_prompt(name),
                tools=sub_agents_tools[name],
                middleware=[
                    SummarizationMiddleware(
                        model=model,
                        trigger=("tokens", 100000),
                        keep=("messages", 15)
                    ),
                    ModelCallLimitMiddleware(
                        thread_limit=10,
                        run_limit=10,
                        exit_behavior="end"
                    )
                ],
            )
            for name in sub_agents_tools
        }

        supervisor = create_agent(
            name="research_supervisor",
            model=model,
            system_prompt=read_prompt("research_supervisor"),
            tools=[handoff_to_multiple_agents_tool] + tools.file_tools,
            checkpointer=checkpointer,  # Will propagate to sub-agents
            middleware=[SummarizationMiddleware(
                model=model,
                trigger=("tokens", 100000),
                keep=("messages", 15)
            )],
        )

        graph = StateGraph(ResearchState)
        graph.add_node(supervisor, destinations=(*sub_agents.keys(), END), defer=True)
        graph.add_node(convert_human_to_ai_messages, defer=False)
        graph.add_edge("convert_human_to_ai_messages", "research_supervisor")
        for name, agent in sub_agents.items():
            graph.add_node(agent)
            graph.add_edge(name, "convert_human_to_ai_messages")

            with open(os.path.join("docs", f"{name}_graph.png"), "wb") as f:
                f.write((await agent.aget_graph()).draw_mermaid_png())

        graph.add_edge(START, "research_supervisor")
        chain = graph.compile(checkpointer=checkpointer)

        with open(os.path.join("docs", "researcher_graph.png"), "wb") as f:
            f.write((await chain.aget_graph()).draw_mermaid_png())

        async for chunk in chain.astream(
            ResearchState(
                messages=[
                    HumanMessage(content="Making modern web cryptography quantum resistant."),
                ],
            ),
            subgraphs=True,
            # stream_mode="debug",
            config={"configurable": {"thread_id": "1"}},
        ):
            # pretty_print_messages(chunk)
            pass


if __name__ == "__main__":
    if os.path.exists("researcher_state.db"):
        os.remove("researcher_state.db")
    if os.path.exists("output"):
        shutil.rmtree("output")
    os.makedirs("output")
    # For now just copy data files to output
    os.system("cp -r data/* output/")
    asyncio.run(main())
