from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from researcher.state import ResearchState


class AgentNode:
    def __init__(self, name: str, model: BaseChatModel, system_prompt: str, tools: list[BaseTool], formatter: type[BaseModel] | None = None, checkpointer: BaseCheckpointSaver | None = None):
        self.__name__ = name
        self.formatter = formatter
        self.formatter_tool: BaseTool | None = tool(formatter) if formatter else None

        self.agent = create_agent(
            name=name,
            model=model,
            tools=tools + ([self.formatter_tool] if self.formatter_tool else []),
            system_prompt=system_prompt,
            checkpointer=checkpointer,
        )
        self.config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    async def __call__(self, state: ResearchState, *args: Any, **kwds: Any) -> Any:
        messages = []
        async for response in self.agent.astream(
            {
                "messages": state["messages"]
            },
            config=self.config,
        ):
            for update in response.values():
                for message in update.get("messages", []):
                    if isinstance(message, AIMessage):
                        messages.append(
                            HumanMessage(content=message.content)
                        )
                    # # else:
                    # messages.append(message)

                    if self.formatter_tool and isinstance(message, ToolMessage) and message.response_metadata["name"] == self.formatter_tool.name and message.content:
                        return {
                            "messages": message.content
                        }

        return {"messages": messages}
