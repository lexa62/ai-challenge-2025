import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from utils.config import settings
from utils.helpers import console

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    console.print("[red]Failed to import MCP SDK[/red]")
    console.print("[yellow]Make sure 'mcp' package is installed: pip install mcp[/yellow]")
    sys.exit(1)


def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com")


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _post_chat_completions(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_seconds: float = 30.0,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = _build_headers(api_key)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _extract_message_content(body: Dict[str, Any]) -> str:
    try:
        return (body["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def _extract_tool_calls(body: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        message = body["choices"][0]["message"]
        if "tool_calls" in message and message["tool_calls"]:
            return message["tool_calls"]
    except Exception:
        pass
    return []


async def list_mcp_tools(session: ClientSession) -> Optional[List[Dict[str, Any]]]:
    """List available tools from MCP server."""
    try:
        result = await session.list_tools()
        if hasattr(result, "tools"):
            return [{"name": tool.name, "description": tool.description, "inputSchema": tool.inputSchema} for tool in result.tools]
        elif isinstance(result, dict) and "tools" in result:
            return result["tools"]
        return None
    except Exception as e:
        console.print(f"[red]Error listing tools: {e}[/red]")
        return None


async def call_mcp_tool(session: ClientSession, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
    """Call a tool on the MCP server."""
    try:
        result = await session.call_tool(tool_name, arguments)
        if hasattr(result, "content"):
            content_parts = []
            for item in result.content:
                if hasattr(item, "text"):
                    content_parts.append(item.text)
                elif isinstance(item, dict) and "text" in item:
                    content_parts.append(item["text"])
            return "\n".join(content_parts) if content_parts else str(result)
        elif isinstance(result, dict):
            if "content" in result:
                content_parts = []
                for item in result["content"]:
                    if isinstance(item, dict) and "text" in item:
                        content_parts.append(item["text"])
                return "\n".join(content_parts) if content_parts else str(result)
            return str(result)
        return str(result)
    except Exception as e:
        console.print(f"[red]Error calling tool {tool_name}: {e}[/red]")
        return None


def convert_mcp_tools_to_openai_format(mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert MCP tool format to OpenAI function calling format."""
    openai_tools = []
    for tool in mcp_tools:
        name = tool.get("name", "")
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", {})

        properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
        required = input_schema.get("required", []) if isinstance(input_schema, dict) else []

        openai_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": required
                }
            }
        }

        for prop_name, prop_schema in properties.items():
            param_type = prop_schema.get("type", "string")
            param_desc = prop_schema.get("description", "")

            openai_tool["function"]["parameters"]["properties"][prop_name] = {
                "type": param_type,
                "description": param_desc
            }

        openai_tools.append(openai_tool)

    return openai_tools


async def run_agent_with_mcp_tools(
    session: ClientSession,
    base_url: str,
    api_key: str,
    model: str,
    user_message: str,
    messages: List[Dict[str, str]],
) -> bool:
    """Run agent with MCP tool calling capability."""
    messages.append({"role": "user", "content": user_message})

    mcp_tools = await list_mcp_tools(session)
    if not mcp_tools:
        console.print("[yellow]No tools available from MCP server[/yellow]")
        return False

    openai_tools = convert_mcp_tools_to_openai_format(mcp_tools)

    payload = {
        "model": model,
        "messages": messages,
        "tools": openai_tools,
        "tool_choice": "auto"
    }

    try:
        body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=30.0)
    except requests.Timeout:
        console.print("[red]Request timed out. Try again or adjust your prompt.[/red]")
        return False
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else "unknown"
        snippet = ""
        try:
            err_json = http_err.response.json() if http_err.response is not None else {}
            snippet = err_json.get("error", {}).get("message", "")
        except Exception:
            try:
                snippet = http_err.response.text[:300] if http_err.response is not None else ""
            except Exception:
                snippet = ""
        console.print(f"[red]HTTP {status}[/red] {snippet}")
        return False
    except requests.RequestException as req_err:
        console.print(f"[red]Network error:[/red] {req_err}")
        return False
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON response from server.[/red]")
        return False

    assistant_message = body["choices"][0]["message"]
    tool_calls = _extract_tool_calls(body)

    if tool_calls:
        messages.append(assistant_message)

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])

            console.print(f"[dim]Calling tool: {tool_name}[/dim]")
            tool_result = await call_mcp_tool(session, tool_name, tool_args)

            if tool_result:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
                console.print(Panel(tool_result, border_style="cyan", title=f"Tool Result: {tool_name}"))

        payload = {
            "model": model,
            "messages": messages,
            "tools": openai_tools,
            "tool_choice": "auto"
        }

        try:
            body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=30.0)
            assistant_message = body["choices"][0]["message"]
            content = _extract_message_content(body)
            if content:
                messages.append(assistant_message)
                md = Markdown(content, code_theme="monokai", justify="left")
                console.print(Panel.fit(md, border_style="green", title="Assistant Response"))
        except Exception as e:
            console.print(f"[red]Error getting final response: {e}[/red]")
            return False
    else:
        content = _extract_message_content(body)
        if content:
            messages.append(assistant_message)
            md = Markdown(content, code_theme="monokai", justify="left")
            console.print(Panel.fit(md, border_style="green", title="Assistant Response"))
        else:
            console.print("[yellow]No content returned.[/yellow]")
            return False

    return True


def display_tools(tools: List[Dict[str, Any]]) -> None:
    """Display available MCP tools in a formatted table."""
    if not tools:
        console.print("[yellow]No tools available[/yellow]")
        return

    console.print()
    console.rule("[bold cyan]Available MCP Tools[/bold cyan]")
    console.print()

    tools_table = Table(show_header=True, header_style="bold magenta")
    tools_table.add_column("Tool Name", style="cyan", width=30)
    tools_table.add_column("Description", style="white")
    tools_table.add_column("Parameters", style="dim", width=40)

    for tool in tools:
        name = tool.get("name", "Unknown")
        description = tool.get("description", "No description available")
        input_schema = tool.get("inputSchema", {})

        properties = {}
        if isinstance(input_schema, dict):
            properties = input_schema.get("properties", {})

        param_info = ", ".join(properties.keys()) if properties else "None"
        if len(param_info) > 35:
            param_info = param_info[:32] + "..."

        tools_table.add_row(name, description, param_info)

    console.print(tools_table)
    console.print()


async def main_async() -> None:
    """Main async function to run the MCP tool demonstration."""
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 9 — Custom MCP Tool[/bold cyan]")
    console.print()
    console.print("[dim]Starting MCP server and connecting agent...[/dim]")
    console.print()

    server_path = Path(__file__).parent / "server.py"
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_path)],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                console.print("[green]✓ Connected to MCP server[/green]")
                console.print()

                tools = await list_mcp_tools(session)
                if tools:
                    display_tools(tools)

                console.print(f"[dim]Model:[/dim] {model}  [dim]Base URL:[/dim] {base_url}")
                console.print("[dim]Type ':q' to quit[/dim]")
                console.print()

                messages: List[Dict[str, str]] = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with access to weather information. When users ask about weather, use the get_weather tool to provide accurate information."
                    }
                ]

                initial_message = "What's the weather like in New York?"
                console.print(f"[dim]Example query: {initial_message}[/dim]")
                console.print()

                await run_agent_with_mcp_tools(
                    session, base_url, api_key, model, initial_message, messages
                )

                console.print()
                console.print("[dim]You can now ask about weather in different cities.[/dim]")
                console.print()

                while True:
                    try:
                        user_input = input("> ").strip()
                    except (EOFError, KeyboardInterrupt):
                        console.print("\n[yellow]Bye.[/yellow]")
                        break

                    if user_input in {"", ":q", ":quit"}:
                        console.print("[yellow]Goodbye![/yellow]")
                        break

                    await run_agent_with_mcp_tools(
                        session, base_url, api_key, model, user_input, messages
                    )
                    console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


def run() -> None:
    """Main entry point for Day 9 task."""
    asyncio.run(main_async())




