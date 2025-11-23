import asyncio
import os
import shlex
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def list_mcp_tools_stdio(server_command: List[str]) -> Optional[Dict[str, Any]]:
    """
    Connect to an MCP server via stdio transport and retrieve available tools.

    Args:
        server_command: Command to run the MCP server (e.g., ["python", "-m", "mcp_server"])

    Returns:
        Dictionary containing tools information, or None if connection fails
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        server_params = StdioServerParameters(
            command=server_command[0],
            args=server_command[1:] if len(server_command) > 1 else [],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                result = await session.list_tools()
                return result

    except ImportError:
        try:
            from mcp.client import Client
            from mcp.client.stdio import StdioClientTransport

            transport = StdioClientTransport(
                command=server_command[0],
                args=server_command[1:] if len(server_command) > 1 else [],
            )

            client = Client(
                name="mcp-client",
                version="1.0.0",
            )

            await client.connect(transport)
            result = await client.list_tools()
            await client.close()
            return result

        except ImportError as e:
            console.print(f"[red]Failed to import MCP SDK: {e}[/red]")
            console.print("[yellow]Make sure 'mcp' package is installed: pip install mcp[/yellow]")
            return None
        except Exception as e:
            console.print(f"[red]Error connecting to MCP server (alternative API): {e}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Error connecting to MCP server: {e}[/red]")
        return None


async def list_mcp_tools_http(server_url: str) -> Optional[Any]:
    """
    Connect to an MCP server via streamable HTTP transport and retrieve available tools.

    Args:
        server_url: URL of the MCP server (e.g., "http://localhost:8000/mcp")

    Returns:
        MCP SDK ListToolsResult object containing tools information, or None if connection fails
    """
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.list_tools()
                return result

    except ImportError as e:
        console.print(f"[red]Failed to import MCP SDK: {e}[/red]")
        console.print("[yellow]Make sure 'mcp' package is installed: pip install mcp[/yellow]")
        return None
    except Exception as e:
        console.print(f"[red]Error connecting to MCP server via HTTP: {e}[/red]")
        return None


def display_tools(tools_result: Any) -> None:
    """
    Display the list of available MCP tools in a formatted table.

    Args:
        tools_result: Tools result from MCP server (can be SDK result object or dict)
    """
    if not tools_result:
        console.print("[yellow]No tools found or invalid response from MCP server[/yellow]")
        return

    tools = None
    if hasattr(tools_result, "tools"):
        tools = tools_result.tools
    elif isinstance(tools_result, dict) and "tools" in tools_result:
        tools = tools_result["tools"]
    else:
        console.print("[yellow]Unexpected response format from MCP server[/yellow]")
        return

    if not tools:
        console.print("[yellow]MCP server returned an empty list of tools[/yellow]")
        return

    console.print()
    console.rule("[bold cyan]Available MCP Tools[/bold cyan]")
    console.print()

    tools_table = Table(show_header=True, header_style="bold magenta")
    tools_table.add_column("Tool Name", style="cyan", width=30)
    tools_table.add_column("Description", style="white")
    tools_table.add_column("Parameters", style="dim", width=40)

    for tool in tools:
        if hasattr(tool, "name"):
            name = tool.name
            description = getattr(tool, "description", "No description available")
            input_schema = getattr(tool, "inputSchema", {})
        elif isinstance(tool, dict):
            name = tool.get("name", "Unknown")
            description = tool.get("description", "No description available")
            input_schema = tool.get("inputSchema", {})
        else:
            continue

        properties = {}
        if isinstance(input_schema, dict):
            properties = input_schema.get("properties", {})
        elif hasattr(input_schema, "properties"):
            properties = input_schema.properties if hasattr(input_schema.properties, "keys") else {}

        param_info = ", ".join(properties.keys()) if properties else "None"

        if len(param_info) > 35:
            param_info = param_info[:32] + "..."

        tools_table.add_row(name, description, param_info)

    console.print(tools_table)
    console.print()
    console.print(f"[dim]Total tools available: {len(tools)}[/dim]")


def run() -> None:
    """
    Main function to connect to MCP server and list available tools.
    """
    console.print("[bold cyan]Day 8 — MCP Integration[/bold cyan]")
    console.print()
    console.print("[dim]This demonstration connects to an MCP server and lists available tools.[/dim]")
    console.print()

    mcp_server_command = os.getenv("MCP_SERVER_COMMAND", "")
    mcp_server_url = os.getenv("MCP_SERVER_URL", "")

    if mcp_server_command:
        console.print(f"[dim]Connecting to MCP server via stdio: {mcp_server_command}[/dim]")
        console.print()

        command_parts = shlex.split(mcp_server_command)
        tools_result = asyncio.run(list_mcp_tools_stdio(command_parts))

    elif mcp_server_url:
        console.print(f"[dim]Connecting to MCP server via HTTP: {mcp_server_url}[/dim]")
        console.print()

        tools_result = asyncio.run(list_mcp_tools_http(mcp_server_url))

    else:
        console.print("[yellow]No MCP server configuration found.[/yellow]")
        console.print()
        console.print("[dim]To use this tool, set one of the following environment variables:[/dim]")
        console.print()
        console.print(Panel(
            "[cyan]MCP_SERVER_COMMAND[/cyan] - Command to run MCP server via stdio\n"
            "  Example: MCP_SERVER_COMMAND='python -m mcp_server'\n\n"
            "[cyan]MCP_SERVER_URL[/cyan] - URL for HTTP-based MCP server (streamable HTTP transport)\n"
            "  Example: MCP_SERVER_URL='http://localhost:8000/mcp'",
            border_style="blue",
            title="Configuration"
        ))
        console.print()
        console.print("[dim]For testing, you can create a simple MCP server or use an existing one.[/dim]")
        return

    if tools_result:
        display_tools(tools_result)
        console.print("[green]✓ Successfully retrieved tools from MCP server[/green]")
    else:
        console.print("[red]✗ Failed to retrieve tools from MCP server[/red]")

