from day01_simple_ai_agent import agent
from utils.helpers import console
from rich.panel import Panel


STRICT_MIN_JSON_PROMPT = (
    "You are an assistant that MUST output ONLY JSON.\n"
    "Produce exactly one JSON object matching this schema:\n"
    '{ "final_answer": string }\n'
    "Rules:\n"
    "1) Output ONLY the JSON object (no prose, no backticks, no markdown).\n"
    "2) Use double quotes for all keys/strings.\n"
    "3) No trailing commas.\n"
    "4) Do not add any extra keys.\n"
    '5) If unsure or unable, return {"final_answer": ""}.\n'
    "\n"
    "Example:\n"
    '{"final_answer": "Paris"}\n'
)


def run() -> None:
    console.print("[cyan]Day 2 — Prompt-only Structured Output[/cyan]")
    console.print('[dim]All responses will be a single JSON object of shape {"final_answer": string}. No prose or markdown.[/dim]')
    console.print(Panel.fit(STRICT_MIN_JSON_PROMPT, border_style="cyan", title="System Prompt"))
    agent.run_cli(system_prompt=STRICT_MIN_JSON_PROMPT)


