import asyncio
import os

import aiosqlite
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from researcher.state import ResearchState
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


async def main():
    async with aiosqlite.connect("researcher_state.db") as conn:
        model = ChatDeepSeek(model="deepseek-reasoner")
        client = MultiServerMCPClient(
            {
                "arxiv": {
                    "transport": "stdio",
                    "command": "uv",
                    "args": [
                        "run", "arxiv-mcp-server", "--storage-path", "./arxiv_data"
                    ],
                }
            }
        )
        mcp_tools = await client.get_tools()
        checkpointer = AsyncSqliteSaver(conn)

        sub_agents_tools = {
            "literature_review_agent_1": mcp_tools,
            "literature_review_agent_2": mcp_tools,
            "proposal_writer_agent": [],
        }

        sub_agents = {
            name: create_agent(
                name=name,
                model=model,
                system_prompt=read_prompt(name),
                tools=sub_agents_tools[name],
                checkpointer=checkpointer,
            )
            for name in sub_agents_tools
        }

        supervisor = create_agent(
            name="research_supervisor",
            model=model,
            system_prompt=read_prompt("research_supervisor"),
            tools=[handoff_to_multiple_agents_tool],
            checkpointer=checkpointer,
        )

        graph = StateGraph(ResearchState)
        graph.add_node(supervisor, destinations=(*sub_agents.keys(), END))
        for name, agent in sub_agents.items():
            graph.add_node(agent)
            graph.add_edge(name, "research_supervisor")

        graph.add_edge(START, "research_supervisor")
        chain = graph.compile(checkpointer=checkpointer)

        with open(os.path.join("docs", "researcher_graph.png"), "wb") as f:
            f.write((await chain.aget_graph()).draw_mermaid_png())

        async for chunk in chain.astream(
            ResearchState(
                ideas=[],
                proposal="",
                bibliography="",
                final_paper="",
                literature_review="",
                messages=[
                    HumanMessage(content="LLM steering")
                ],
            ),
            subgraphs=True,
            # stream_mode="debug",
            config={"configurable": {"thread_id": "1"}},
        ):
            pretty_print_messages(chunk)

        # Save final graph state into text file
        final_state = chain.get_state({"configurable": {"thread_id": "1"}})
        with open("final_research_state.txt", "w") as f:
            f.write(str(final_state))


if __name__ == "__main__":
    if os.path.exists("researcher_state.db"):
        os.remove("researcher_state.db")
    asyncio.run(main())
