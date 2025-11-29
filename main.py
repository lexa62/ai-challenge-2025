import argparse
import importlib
import sys
from typing import Optional, Union

from utils.helpers import console


DAY_NUM_TO_PACKAGE = {
    1: "day01_simple_ai_agent",
    2: "day02_ai_configuration",
    3: "day03_interaction",
    4: "day04_temperature",
    5: "day05_tokens",
    6: "day06_subagents_interaction",
    7: "day07_dialogue_compression",
    8: "day08_mcp_integration",
    9: "day09_custom_mcp_tool",
    10: "day10_mcp_tools_composition",
    11: "day11_external_memory",
    12: "day12_voice_agent",
    13: "day13_document_indexing",
}

NAME_TO_DAY_NUM = {
    "day01": 1,
    "day1": 1,
    "prompting": 1,
    "simple_ai_agent": 1,
    "day02": 2,
    "day2": 2,
    "ai_configuration": 2,
    "day03": 3,
    "day3": 3,
    "interaction": 3,
    "day04": 4,
    "day4": 4,
    "temperature": 4,
    "day05": 5,
    "day5": 5,
    "tokens": 5,
    "token_counting": 5,
    "day06": 6,
    "day6": 6,
    "subagents": 6,
    "subagents_interaction": 6,
    "day07": 7,
    "day7": 7,
    "dialogue_compression": 7,
    "compression": 7,
    "day08": 8,
    "day8": 8,
    "mcp_integration": 8,
    "mcp": 8,
    "day09": 9,
    "day9": 9,
    "custom_mcp_tool": 9,
    "custom_mcp": 9,
    "day10": 10,
    "mcp_tools_composition": 10,
    "tools_composition": 10,
    "mcp_composition": 10,
    "day11": 11,
    "day11_external_memory": 11,
    "external_memory": 11,
    "memory": 11,
    "day12": 12,
    "day12_voice_agent": 12,
    "voice_agent": 12,
    "voice": 12,
    "day13": 13,
    "day13_document_indexing": 13,
    "document_indexing": 13,
    "indexing": 13,
}


def resolve_day_identifier(day: Union[int, str]) -> Optional[int]:
    if isinstance(day, int):
        return day if day in DAY_NUM_TO_PACKAGE else None
    text = str(day).strip().lower()
    if text.isdigit():
        num = int(text)
        return num if num in DAY_NUM_TO_PACKAGE else None
    return NAME_TO_DAY_NUM.get(text)


def run_task(day: Union[int, str]) -> None:
    num = resolve_day_identifier(day)
    if num is None:
        console.print(f"[red]Unknown day/task:[/red] {day}")
        sys.exit(1)
    package = DAY_NUM_TO_PACKAGE[num]
    module_name = f"{package}.task"
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        console.print(f"[red]Failed to import module[/red] {module_name}: {exc}")
        sys.exit(1)
    if not hasattr(module, "run"):
        console.print(f"[red]Module {module_name} has no 'run' function.[/red]")
        sys.exit(1)
    module.run()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="AI Advent Challenge 2025 CLI")
    parser.add_argument("--day", type=int, help="Run by day number (e.g., 1)")
    parser.add_argument("--task", type=str, help="Run by task name (e.g., day01, prompting)")
    args = parser.parse_args(argv)

    if args.day is not None:
        run_task(args.day)
        return
    if args.task:
        run_task(args.task)
        return

    # Default behavior: run Day 1 agent directly
    run_task(1)


if __name__ == "__main__":
    main()

