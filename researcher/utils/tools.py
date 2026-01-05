import base64
import json
import os
from subprocess import PIPE, Popen
from typing import Annotated

from dotenv import load_dotenv
from langchain.messages import ToolMessage
from langchain.tools import BaseTool, tool
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import (FileManagementToolkit,
                                                PlayWrightBrowserToolkit)
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.tools import BraveSearch
from langchain_community.tools.playwright.utils import \
    create_async_playwright_browser
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from researcher.utils import loader

load_dotenv()

OUTPUT_DIR = os.path.join(os.getcwd(), "output")


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

    return tool(wrapper)


@tool
def read_binary_file(file_path: str) -> dict:
    """Reads a binary file and returns its content encoded in base64."""
    path = os.path.join(OUTPUT_DIR, file_path)

    if not os.path.exists(path):
        return {
            "type": "input_text",
            "content": f"File {path} does not exist."
        }

    with open(path, "rb") as f:
        encoded_content = base64.b64encode(f.read())

    # return encoded_content
    return {
        "type": "input_file",
        "filename": os.path.basename(path),
        "file_data": encoded_content.decode("utf-8"),
    }


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

file_tools = file_toolkit.get_tools() + [read_binary_file]

brave_search_tool = BraveSearch()

request_tools = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=True
).get_tools()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=256)
chroma_store = Chroma(
    collection_name="researcher_collection",
    embedding_function=embeddings,
    persist_directory="./"
)
documents = loader.load_documents("data")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
docs = text_splitter.split_documents(documents)
chroma_store.add_documents(docs)


@tool(response_format="content_and_artifact")
def retrieve_relevant_documents(
    query: str,
    k: Annotated[int, "The number of relevant documents to retrieve from the vector store."] = 5
) -> dict:
    """Retrieve relevant documents from the vector store based on the query."""
    results = chroma_store.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in results
    )
    return serialized, results


@tool
def run_typst_command(
    argv: Annotated[list[str], "Arguments to pass to the Typst command."]
) -> str:
    """
    Run a Typst command with the given arguments and return the output.
    You can use this tool to generate documents in Typst format, or check for errors in Typst files.
    Example usage: argv = ["compile", "document.typ"]
    """
    process = Popen(["typst", *argv], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()

    return stdout.decode("utf-8") + "\n" + stderr.decode("utf-8")


@tool
def load_url_content(
    urls: Annotated[list[str], "A list of URLs to load content from. Include the full URL with http/https."],
) -> str:
    """Load the content of multiple URLs using a browser."""
    loader = SeleniumURLLoader(
        urls=urls,
        headless=False,
        browser="chrome"
    )
    documents = loader.load()

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in documents
    )

    return serialized
