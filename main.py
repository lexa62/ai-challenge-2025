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
    14: "day14_rag_query",
    15: "day15_reranking_filtering",
    16: "day16_citations_sources",
    17: "day17_image_generation",
    18: "day18_prompt_style_systems",
    19: "day19_vision_qa_agent",
    20: "day20_meeting_summarizer",
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
    "day14": 14,
    "day14_rag_query": 14,
    "rag_query": 14,
    "rag": 14,
    "day15": 15,
    "day15_reranking_filtering": 15,
    "reranking_filtering": 15,
    "reranking": 15,
    "filtering": 15,
    "day16": 16,
    "day16_citations_sources": 16,
    "citations_sources": 16,
    "citations": 16,
    "sources": 16,
    "day17": 17,
    "day17_image_generation": 17,
    "image_generation": 17,
    "image_gen": 17,
    "day18": 18,
    "day18_prompt_style_systems": 18,
    "prompt_style_systems": 18,
    "style_systems": 18,
    "prompt_style": 18,
    "day19": 19,
    "day19_vision_qa_agent": 19,
    "vision_qa_agent": 19,
    "vision_qa": 19,
    "qa_agent": 19,
    "day20": 20,
    "day20_meeting_summarizer": 20,
    "meeting_summarizer": 20,
    "meeting_notes": 20,
}


def resolve_day_identifier(day: Union[int, str]) -> Optional[int]:
    if isinstance(day, int):
        return day if day in DAY_NUM_TO_PACKAGE else None
    text = str(day).strip().lower()
    if text.isdigit():
        num = int(text)
        return num if num in DAY_NUM_TO_PACKAGE else None
    return NAME_TO_DAY_NUM.get(text)


def run_task(day: Union[int, str], extra_args: Optional[list[str]] = None) -> None:
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

    if extra_args:
        # Prefer tasks that accept an argv-like parameter; fall back to legacy signature.
        try:
            module.run(extra_args)
            return
        except TypeError:
            original_argv = sys.argv
            try:
                sys.argv = [original_argv[0]] + list(extra_args)
                module.run()
            finally:
                sys.argv = original_argv
    else:
        module.run()


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="AI Advent Challenge 2025 CLI")
    parser.add_argument("--day", type=int, help="Run by day number (e.g., 1)")
    parser.add_argument("--task", type=str, help="Run by task name (e.g., day01, prompting)")
    parser.add_argument(
        "task_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to the selected task.",
    )
    args = parser.parse_args(argv)

    extra_args = args.task_args or None

    if args.day is not None:
        run_task(args.day, extra_args=extra_args)
        return
    if args.task:
        run_task(args.task, extra_args=extra_args)
        return

    # Default behavior: run Day 1 agent directly
    run_task(1)


if __name__ == "__main__":
    main()

