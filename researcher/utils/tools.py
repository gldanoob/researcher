import json

from langchain.tools import BaseTool, tool
from pydantic import BaseModel


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
