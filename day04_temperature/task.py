import json
import os
from typing import Any, Dict, List, Optional

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from utils.config import settings

console = Console()

TEMPERATURES = [0.0, 0.7, 1.2]

TEST_PROMPTS = [
    {
        "name": "Factual/Accuracy",
        "prompt": "What is the capital of France?",
        "description": "Tests consistency and accuracy for factual questions",
    },
    {
        "name": "Creative Writing",
        "prompt": "Write a short story about a robot learning to paint (2-3 sentences).",
        "description": "Tests creativity and variation in storytelling",
    },
    {
        "name": "Code Generation",
        "prompt": "Write a Python function to calculate fibonacci numbers.",
        "description": "Tests technical accuracy vs creative solutions",
    },
    {
        "name": "Problem Solving",
        "prompt": "Explain how photosynthesis works in 2-3 sentences.",
        "description": "Tests structured vs creative explanations",
    },
]


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com")


def _build_payload(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
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


def run_prompt_with_temperature(
    prompt: str,
    temperature: float,
    model: str,
    base_url: str,
    api_key: str,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    payload = _build_payload(model, messages, temperature)

    try:
        body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=30.0)
        return _extract_message_content(body)
    except requests.Timeout:
        return "[Error: Request timed out]"
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else "unknown"
        return f"[Error: HTTP {status}]"
    except requests.RequestException as req_err:
        return f"[Error: Network error - {req_err}]"
    except json.JSONDecodeError:
        return "[Error: Invalid JSON response]"
    except Exception as e:
        return f"[Error: {str(e)}]"


def analyze_responses(responses: Dict[float, str], prompt_type: str) -> Dict[str, Any]:
    analysis = {
        "accuracy": {},
        "creativity": {},
        "diversity": {},
    }

    temp_0 = responses.get(0.0, "")
    temp_07 = responses.get(0.7, "")
    temp_12 = responses.get(1.2, "")

    if prompt_type == "Factual/Accuracy":
        analysis["accuracy"][0.0] = "High - Deterministic, consistent answer"
        analysis["accuracy"][0.7] = "High - Usually correct, slight variation possible"
        analysis["accuracy"][1.2] = "Medium - May introduce creative but incorrect details"

        analysis["creativity"][0.0] = "None - Same answer every time"
        analysis["creativity"][0.7] = "Low - Minor variations in phrasing"
        analysis["creativity"][1.2] = "Medium - May add creative context"

        same_content = temp_0 == temp_07 == temp_12
        similar_content = temp_0 == temp_07 or temp_07 == temp_12
        analysis["diversity"]["overall"] = "Low diversity" if same_content else ("Medium diversity" if similar_content else "High diversity")

    elif prompt_type == "Creative Writing":
        analysis["accuracy"][0.0] = "N/A - Creative task"
        analysis["accuracy"][0.7] = "N/A - Creative task"
        analysis["accuracy"][1.2] = "N/A - Creative task"

        analysis["creativity"][0.0] = "Low - Repetitive, formulaic"
        analysis["creativity"][0.7] = "Medium - Balanced creativity"
        analysis["creativity"][1.2] = "High - More varied and original"

        len_0 = len(temp_0)
        len_07 = len(temp_07)
        len_12 = len(temp_12)
        variance = max(len_0, len_07, len_12) - min(len_0, len_07, len_12)
        analysis["diversity"]["overall"] = "High diversity" if variance > 50 else ("Medium diversity" if variance > 20 else "Low diversity")

    elif prompt_type == "Code Generation":
        analysis["accuracy"][0.0] = "High - Most reliable, correct implementation"
        analysis["accuracy"][0.7] = "High - Usually correct, may have minor variations"
        analysis["accuracy"][1.2] = "Medium - May include creative but potentially buggy solutions"

        analysis["creativity"][0.0] = "Low - Standard implementation"
        analysis["creativity"][0.7] = "Medium - Some variation in approach"
        analysis["creativity"][1.2] = "High - More creative solutions, may be unconventional"

        has_errors_12 = "Error" in temp_12 or "error" in temp_12.lower()
        analysis["diversity"]["overall"] = "High diversity" if has_errors_12 else "Medium diversity"

    else:
        analysis["accuracy"][0.0] = "High - Structured, factual"
        analysis["accuracy"][0.7] = "High - Balanced accuracy and readability"
        analysis["accuracy"][1.2] = "Medium - May include creative analogies or less precise language"

        analysis["creativity"][0.0] = "Low - Straightforward explanation"
        analysis["creativity"][0.7] = "Medium - Engaging but accurate"
        analysis["creativity"][1.2] = "High - Creative analogies, varied explanations"

        analysis["diversity"]["overall"] = "Medium to High diversity"

    return analysis


def display_comparison(
    prompt_info: Dict[str, str],
    responses: Dict[float, str],
    analysis: Dict[str, Any],
) -> None:
    console.print()
    console.rule(f"[bold cyan]{prompt_info['name']}[/bold cyan]")
    console.print(f"[dim]{prompt_info['description']}[/dim]")
    console.print()

    console.print(f"[bold]Prompt:[/bold] {prompt_info['prompt']}")
    console.print()

    for temp in TEMPERATURES:
        response = responses.get(temp, "[No response]")
        color = "blue" if temp == 0.0 else ("yellow" if temp == 0.7 else "red")
        title = f"Temperature {temp}"

        md = Markdown(response, code_theme="monokai")
        console.print(Panel(md, border_style=color, title=title))
        console.print()

    console.print("[bold]Analysis:[/bold]")
    analysis_table = Table(show_header=True, header_style="bold magenta")
    analysis_table.add_column("Temperature", style="cyan")
    analysis_table.add_column("Accuracy", style="green")
    analysis_table.add_column("Creativity", style="yellow")

    for temp in TEMPERATURES:
        acc = analysis["accuracy"].get(temp, "N/A")
        creat = analysis["creativity"].get(temp, "N/A")
        analysis_table.add_row(str(temp), acc, creat)

    console.print(analysis_table)
    console.print(f"[dim]Diversity: {analysis['diversity'].get('overall', 'N/A')}[/dim]")
    console.print()


def run_temperature_comparison() -> None:
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 4 — Temperature Comparison[/bold cyan]")
    console.print()
    console.print("[dim]This demonstration runs the same prompts with three different temperature values[/dim]")
    console.print("[dim](0, 0.7, and 1.2) to show how temperature affects model outputs.[/dim]")
    console.print()
    console.print(f"[dim]Model:[/dim] {model}  [dim]Base URL:[/dim] {base_url}")
    console.print()

    for prompt_info in TEST_PROMPTS:
        console.print(f"[yellow]Running: {prompt_info['name']}...[/yellow]")

        responses: Dict[float, str] = {}
        for temp in TEMPERATURES:
            response = run_prompt_with_temperature(
                prompt_info["prompt"],
                temp,
                model,
                base_url,
                api_key,
            )
            responses[temp] = response

        analysis = analyze_responses(responses, prompt_info["name"])
        display_comparison(prompt_info, responses, analysis)

    console.rule("[bold cyan]Summary & Recommendations[/bold cyan]")
    console.print()

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Task Type", style="cyan")
    summary_table.add_column("Best Temperature", style="green")
    summary_table.add_column("Reason", style="white")

    summary_table.add_row(
        "Factual/Accuracy",
        "0.0",
        "Deterministic, consistent answers for factual queries"
    )
    summary_table.add_row(
        "Creative Writing",
        "0.7 - 1.2",
        "Higher temperature increases creativity and variation"
    )
    summary_table.add_row(
        "Code Generation",
        "0.0 - 0.7",
        "Lower temperature ensures correctness and reliability"
    )
    summary_table.add_row(
        "Problem Solving",
        "0.7",
        "Balanced: accurate but engaging explanations"
    )

    console.print(summary_table)
    console.print()
    console.print("[bold]General Guidelines:[/bold]")
    console.print("  • Temperature 0.0: Use for factual queries, code generation, and when consistency is critical")
    console.print("  • Temperature 0.7: Good default for most tasks, balanced creativity and accuracy")
    console.print("  • Temperature 1.2: Use for creative writing, brainstorming, and when diversity is desired")
    console.print()


def run() -> None:
    run_temperature_comparison()

