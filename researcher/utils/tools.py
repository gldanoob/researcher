import json
import os

from langchain.tools import BaseTool, tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import BraveSearch
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import CallToolResult, ContentBlock, TextContent
from pydantic import BaseModel

from langchain_community.utilities.requests import TextRequestsWrapper

from dotenv import load_dotenv

load_dotenv()

def structured_output_tool(model: type[BaseModel]) -> BaseTool:
    """Decorator to create a structured output tool from a Pydantic model."""
    def wrapper(**kwargs) -> str:
        return json.dumps(kwargs, indent=2)

    wrapper.__doc__ = model.__doc__ or ""

    field_desc = []
    for field_name, field in model.model_json_schema()["properties"].items():
        field_desc.append(f"{field_name}: {field.get('description', '')}")

    wrapper.__name__ = model.__name__
    wrapper.__doc__ += "\n\n" + "\n".join(field_desc)
    # print(wrapper.__doc__)

    return tool(wrapper)


async def mcp_exception_handler(request: MCPToolCallRequest, handler):
    try:
        response = await handler(request)
        return response
    except Exception as e:
        return CallToolResult(content=[
            TextContent(type="text", text=f"An error occurred while calling the tool: {str(e)}")
        ], isError=True)


async def get_arxiv_tools() -> list[BaseTool]:
    client = MultiServerMCPClient(
        {
            "arxiv": {
                "transport": "stdio",
                "command": "sh",
                "args": [
                    "run.sh"
                ],
            },
        },
        tool_interceptors=[mcp_exception_handler]

    )
    mcp_tools = await client.get_tools()
    return mcp_tools

file_toolkit = FileManagementToolkit(
    root_dir=os.path.join(os.getcwd(), "output")
)
file_tools = file_toolkit.get_tools()

brave_search_tool = BraveSearch()

request_tools = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=True
).get_tools()
