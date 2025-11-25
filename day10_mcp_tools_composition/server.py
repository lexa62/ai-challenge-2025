import os
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Documentation Pipeline Service")

MOCK_DOCUMENTATION = {
    "python async": """
Python Asynchronous Programming

Asynchronous programming in Python allows you to write concurrent code that can handle multiple operations efficiently. The main components are:

1. async/await syntax: Introduced in Python 3.5, this allows you to define asynchronous functions and coroutines.

2. asyncio module: Provides the event loop and utilities for running asynchronous code.

Example:
    import asyncio

    async def fetch_data(url):
        # Simulate network request
        await asyncio.sleep(1)
        return f"Data from {url}"

    async def main():
        results = await asyncio.gather(
            fetch_data("url1"),
            fetch_data("url2")
        )
        return results

    asyncio.run(main())

3. Benefits: Non-blocking I/O operations, better resource utilization, and improved performance for I/O-bound tasks.

4. Common patterns: Using async with for context managers, async for for iterating over async iterables, and asyncio.gather() for concurrent execution.
""",
    "fastapi": """
FastAPI - Modern Python Web Framework

FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

Key Features:
- Fast: Very high performance, on par with NodeJS and Go
- Easy: Designed to be easy to use and learn
- Standards-based: Based on open standards like OpenAPI and JSON Schema
- Automatic documentation: Interactive API docs (Swagger UI and ReDoc)

Example:
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    @app.get("/items/{item_id}")
    def read_item(item_id: int, q: Optional[str] = None):
        return {"item_id": item_id, "q": q}

FastAPI uses Pydantic for data validation and Starlette for the web parts.
""",
    "pydantic": """
Pydantic - Data Validation Library

Pydantic is a data validation library for Python that uses Python type annotations to validate data.

Key Features:
- Type validation: Automatically validates data types
- Data parsing: Converts data to the correct types
- IDE support: Great editor support with type hints
- JSON Schema: Can generate JSON schemas from models

Example:
    from pydantic import BaseModel

    class User(BaseModel):
        name: str
        age: int
        email: str

    user = User(name="John", age=30, email="john@example.com")
    print(user.json())  # {"name": "John", "age": 30, "email": "john@example.com"}
""",
    "mcp": """
Model Context Protocol (MCP)

MCP is a protocol for connecting AI applications with external data sources and tools. It enables:

1. Tool Integration: Connect AI agents with external tools and services
2. Data Access: Access structured data from various sources
3. Standardized Interface: Common protocol for tool and data integration

Key Concepts:
- Servers: Provide tools and resources
- Clients: Consume tools and resources
- Tools: Executable functions that can be called by AI agents
- Resources: Read-only data sources

MCP uses JSON-RPC for communication and supports stdio and HTTP transports.
""",
}


def _search_mock_docs(query: str) -> str:
    """Search mock documentation store."""
    query_lower = query.lower()

    for key, content in MOCK_DOCUMENTATION.items():
        if key in query_lower or query_lower in key:
            return content.strip()

    return f"No documentation found for query: {query}"


@mcp.tool()
def searchDocs(query: str) -> str:
    """
    Search documentation and return relevant content.

    This tool simulates a web search for documentation. It searches through
    a mock documentation store and returns matching content.

    Args:
        query: Search query string (e.g., "python async", "fastapi", "pydantic", "mcp")

    Returns:
        Found documentation content as text
    """
    try:
        result = _search_mock_docs(query)
        return result
    except Exception as e:
        return f"Error searching documentation: {str(e)}"


@mcp.tool()
def summarize(text: str, max_length: Optional[int] = None) -> str:
    """
    Summarize text content using an LLM.

    This tool uses the OpenAI API to generate a concise summary of the provided text.

    Args:
        text: Text content to summarize
        max_length: Optional maximum length for the summary in words (default: auto)

    Returns:
        Summary text
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not set. Cannot summarize text."

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        prompt = f"Please provide a concise summary of the following text:\n\n{text}"
        if max_length:
            prompt += f"\n\nKeep the summary under {max_length} words."

        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        summary = result["choices"][0]["message"]["content"].strip()
        return summary

    except requests.RequestException as e:
        return f"Error calling OpenAI API: {str(e)}"
    except Exception as e:
        return f"Error summarizing text: {str(e)}"


@mcp.tool()
def saveToFile(content: str, filename: str) -> str:
    """
    Save content to a file.

    This tool saves the provided content to a file in the outputs directory.
    The outputs directory will be created if it doesn't exist.

    Args:
        content: Content to save to file
        filename: Output filename (will be saved in outputs/ directory)

    Returns:
        Success message with file path
    """
    try:
        base_dir = Path(__file__).parent.parent
        outputs_dir = base_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        file_path = outputs_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully saved content to {file_path}"

    except Exception as e:
        return f"Error saving file: {str(e)}"


if __name__ == "__main__":
    mcp.run()

