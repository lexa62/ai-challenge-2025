from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel


console = Console()


def header(text: str) -> None:
    console.rule(f"[bold cyan]{text}")


def success(text: str) -> None:
    console.print(Panel.fit(f"[bold green]{text}"))


def info(text: str) -> None:
    console.print(f"[blue]{text}")


def warn(text: str) -> None:
    console.print(f"[yellow]{text}")

