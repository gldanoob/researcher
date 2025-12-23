import json

import dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from researcher.utils.tools import structured_output_tool

dotenv.load_dotenv()

SYSTEM_PROMPT = """You are an advanced deep research agent specializing in literature review of academic papers.
You have access to the Arxiv library. Your task is to assist users in finding and summarizing relevant academic papers based on their queries.
Use the output formatting tool to structure your responses appropriately. Make sure you respond with the tool call only.
"""
config: RunnableConfig = {"configurable": {"thread_id": "1"}}


@structured_output_tool
class LiteratureReviewModel(BaseModel):
    """Formats the literature review response."""

    paper_titles: list[str] = Field(
        description="A list of titles of the relevant papers."
    )
    summaries: list[str] = Field(
        description="A list of summaries corresponding to each paper."
    )
    references: list[str] = Field(
        description="A list of references or links to the papers."
    )
    keywords: list[str] = Field(
        description="A list of keywords related to the research topic."
    )


checkpointer = InMemorySaver()

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


model = ChatDeepSeek(model="deepseek-reasoner")


async def literature_review(prompt: str):
    mcp_tools = await client.get_tools()

    agent = create_agent(
        name="LiteratureReviewAgent",
        model=model,
        tools=mcp_tools + [LiteratureReviewModel],
        checkpointer=checkpointer,
        system_prompt=SYSTEM_PROMPT,
    )

    async for response in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        config=config,
    ):
        for update in response.values():
            for message in update.get("messages", []):
                if isinstance(message, AIMessage):
                    message.pretty_print()
