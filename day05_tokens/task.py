import json
import os
from typing import Any, Dict, List, Optional

import requests
import tiktoken
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from utils.config import settings

console = Console()

MODEL_CONTEXT_LIMITS = {
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
}

MODEL_ENCODINGS = {
    "gpt-4o-mini": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4-turbo": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


def get_model_encoding(model: str) -> str:
    """Map model name to tiktoken encoding name."""
    encoding_name = MODEL_ENCODINGS.get(model, "cl100k_base")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return encoding_name
    except Exception:
        return "cl100k_base"


def count_tokens(text: str, model: str) -> int:
    """Count tokens in text using tiktoken."""
    encoding_name = get_model_encoding(model)
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        console.print(f"[yellow]Warning: Could not count tokens for model {model}: {e}[/yellow]")
        return len(text.split())


def count_messages_tokens(messages: List[Dict[str, str]], model: str) -> int:
    """Count tokens for a list of messages (including formatting overhead)."""
    encoding_name = get_model_encoding(model)
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens
    except Exception as e:
        console.print(f"[yellow]Warning: Could not count tokens for messages: {e}[/yellow]")
        total_text = " ".join([str(msg.get("content", "")) for msg in messages])
        return len(total_text.split())


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
    temperature: float = 0.7,
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
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = _build_headers(api_key)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _extract_usage(body: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract token usage from API response."""
    try:
        usage = body.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception:
        return None


def _extract_message_content(body: Dict[str, Any]) -> str:
    try:
        return (body["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def run_prompt_with_token_counting(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Dict[str, Any]:
    """Run a prompt and return token statistics."""
    messages = [{"role": "user", "content": prompt}]

    estimated_tokens = count_messages_tokens(messages, model)

    payload = _build_payload(model, messages)

    result = {
        "prompt": prompt,
        "estimated_tokens": estimated_tokens,
        "success": False,
        "response_content": "",
        "usage": None,
        "error": None,
    }

    try:
        body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=120.0)
        usage = _extract_usage(body)
        content = _extract_message_content(body)

        result["success"] = True
        result["response_content"] = content
        result["usage"] = usage

    except requests.Timeout:
        result["error"] = "Request timed out"
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else "unknown"
        error_msg = ""
        is_context_limit_error = False
        is_rate_limit_error = False
        try:
            err_json = http_err.response.json() if http_err.response is not None else {}
            error_msg = err_json.get("error", {}).get("message", "")
            error_msg_lower = error_msg.lower()
            is_context_limit_error = (
                "context_length" in error_msg_lower or
                "maximum context length" in error_msg_lower or
                "context window" in error_msg_lower or
                status == 400 and ("context" in error_msg_lower or "length" in error_msg_lower)
            )
            is_rate_limit_error = (
                status == 429 and (
                    "tpm" in error_msg_lower or
                    "tokens per min" in error_msg_lower or
                    "rate limit" in error_msg_lower or
                    "request too large" in error_msg_lower
                )
            )
        except Exception:
            try:
                error_msg = http_err.response.text[:300] if http_err.response is not None else ""
            except Exception:
                error_msg = ""

        if is_context_limit_error:
            result["error"] = f"Context limit exceeded: {error_msg}"
        elif is_rate_limit_error:
            result["error"] = f"Rate limit / Request too large: {error_msg}"
        else:
            result["error"] = f"HTTP {status}: {error_msg}"
    except requests.RequestException as req_err:
        result["error"] = f"Network error: {req_err}"
    except json.JSONDecodeError:
        result["error"] = "Invalid JSON response"
    except Exception as e:
        result["error"] = f"Error: {str(e)}"

    return result


def create_long_prompt(target_tokens: int, model: str) -> str:
    """Create a prompt with approximately target_tokens tokens."""
    base_text = (
        "Please provide a detailed explanation of the following topic: "
        "The history and evolution of artificial intelligence, including key milestones, "
        "major breakthroughs, influential researchers, and the current state of the field. "
        "Include information about machine learning, deep learning, neural networks, "
        "natural language processing, computer vision, and recent developments in large language models. "
    )

    base_tokens = count_tokens(base_text, model)
    if base_tokens == 0:
        base_text = "A" * 100
        base_tokens = count_tokens(base_text, model)
        if base_tokens == 0:
            base_tokens = 1

    repetitions = max(1, target_tokens // base_tokens)
    prompt = base_text * repetitions

    actual_tokens = count_tokens(prompt, model)

    if actual_tokens < target_tokens:
        remaining_tokens = target_tokens - actual_tokens
        additional_repetitions = max(1, remaining_tokens // base_tokens)
        prompt += base_text * additional_repetitions

        final_tokens = count_tokens(prompt, model)
        if final_tokens < target_tokens:
            padding = " " * (target_tokens - final_tokens)
            prompt += padding

    return prompt


TEST_CASES = [
    {
        "name": "Short Prompt",
        "description": "A simple, concise question",
        "prompt": "What is the capital of France?",
    },
    {
        "name": "Long Prompt",
        "description": "A detailed request within context limits",
        "prompt": None,
        "target_tokens": 5000,
    },
    {
        "name": "Exceeding Context Limit",
        "description": "A prompt that exceeds the model's context window",
        "prompt": None,
        "target_tokens": None,
    },
]


def display_token_comparison(
    test_case: Dict[str, Any],
    result: Dict[str, Any],
    model: str,
    context_limit: int,
) -> None:
    """Display token comparison results using Rich."""
    console.print()
    console.rule(f"[bold cyan]{test_case['name']}[/bold cyan]")
    console.print(f"[dim]{test_case['description']}[/dim]")
    console.print()

    if test_case["name"] == "Short Prompt":
        console.print(f"[bold]Prompt:[/bold] {test_case['prompt']}")
    else:
        prompt_preview = result["prompt"][:200] + "..." if len(result["prompt"]) > 200 else result["prompt"]
        console.print(f"[bold]Prompt (preview):[/bold] {prompt_preview}")
        console.print(f"[dim]Full prompt length: {len(result['prompt'])} characters[/dim]")
    console.print()

    token_table = Table(show_header=True, header_style="bold magenta")
    token_table.add_column("Metric", style="cyan")
    token_table.add_column("Value", style="white")

    estimated = result["estimated_tokens"]
    token_table.add_row("Estimated Tokens (tiktoken)", str(estimated))

    if result["usage"]:
        actual_prompt = result["usage"]["prompt_tokens"]
        actual_completion = result["usage"]["completion_tokens"]
        actual_total = result["usage"]["total_tokens"]

        token_table.add_row("Actual Prompt Tokens (API)", str(actual_prompt))
        token_table.add_row("Actual Completion Tokens (API)", str(actual_completion))
        token_table.add_row("Actual Total Tokens (API)", str(actual_total))

        diff = abs(estimated - actual_prompt)
        token_table.add_row("Estimation Difference", f"{diff} tokens ({diff/actual_prompt*100:.1f}%)" if actual_prompt > 0 else "N/A")
    else:
        token_table.add_row("Actual Tokens (API)", "[red]N/A (request failed)[/red]")

    token_table.add_row("Model Context Limit", f"{context_limit:,} tokens")

    if result["usage"]:
        usage_percent = (result["usage"]["total_tokens"] / context_limit) * 100
        token_table.add_row("Context Usage", f"{usage_percent:.2f}%")

    console.print(token_table)
    console.print()

    if result["success"]:
        if result["response_content"]:
            content_preview = result["response_content"][:500]
            if len(result["response_content"]) > 500:
                content_preview += "..."
            md = Markdown(content_preview, code_theme="monokai")
            console.print(Panel(md, border_style="green", title="Response (preview)"))
        else:
            console.print("[yellow]No content in response[/yellow]")
    else:
        error_msg = result['error']
        if "rate limit" in error_msg.lower() or "tpm" in error_msg.lower() or "request too large" in error_msg.lower():
            console.print(Panel(
                f"[red]{error_msg}[/red]\n\n"
                f"[yellow]Note:[/yellow] This error indicates the request exceeded rate limits (TPM). "
                f"For very large prompts, rate limits may be hit before context limits are reached. "
                f"The prompt size ({estimated:,} tokens) exceeds the context limit ({context_limit:,} tokens), "
                f"demonstrating that the model cannot process requests of this size.",
                border_style="red",
                title="Error: Request Too Large"
            ))
        else:
            console.print(Panel(f"[red]{error_msg}[/red]", border_style="red", title="Error"))

    console.print()


def analyze_results(results: List[Dict[str, Any]], model: str, context_limit: int) -> None:
    """Analyze and display summary of all results."""
    console.print()
    console.rule("[bold cyan]Summary & Analysis[/bold cyan]")
    console.print()

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Test Case", style="cyan")
    summary_table.add_column("Estimated Tokens", style="white")
    summary_table.add_column("Actual Tokens", style="white")
    summary_table.add_column("Status", style="white")
    summary_table.add_column("Behavior", style="white")

    for i, result in enumerate(results):
        test_case = TEST_CASES[i]
        estimated = result["estimated_tokens"]

        if result["usage"]:
            actual = result["usage"]["total_tokens"]
            status = "[green]Success[/green]"
            if actual > context_limit * 0.9:
                behavior = "[yellow]Near limit[/yellow]"
            elif actual > context_limit:
                behavior = "[red]Exceeded limit[/red]"
            else:
                behavior = "[green]Normal[/green]"
        else:
            actual = "N/A"
            status = "[red]Failed[/red]"
            if estimated > context_limit:
                error_lower = (result.get("error") or "").lower()
                if "rate limit" in error_lower or "tpm" in error_lower or "request too large" in error_lower:
                    behavior = "[red]Exceeded limit (rate limit hit)[/red]"
                else:
                    behavior = "[red]Exceeded limit (expected)[/red]"
            else:
                behavior = "[yellow]Unexpected error[/yellow]"

        summary_table.add_row(
            test_case["name"],
            str(estimated),
            str(actual),
            status,
            behavior,
        )

    console.print(summary_table)
    console.print()

    console.print("[bold]Key Observations:[/bold]")
    console.print()

    short_result = results[0]
    long_result = results[1]
    exceed_result = results[2]

    if short_result["usage"]:
        console.print("  • [green]Short prompt:[/green] Works normally with minimal token usage")
        console.print(f"    - Estimated: {short_result['estimated_tokens']} tokens")
        console.print(f"    - Actual: {short_result['usage']['total_tokens']} tokens")
        console.print()

    if long_result["usage"]:
        console.print("  • [yellow]Long prompt:[/yellow] Handles large prompts within limits")
        console.print(f"    - Estimated: {long_result['estimated_tokens']} tokens")
        console.print(f"    - Actual: {long_result['usage']['total_tokens']} tokens")
        console.print(f"    - Context usage: {(long_result['usage']['total_tokens'] / context_limit) * 100:.2f}%")
        console.print()
    elif long_result["error"]:
        console.print("  • [yellow]Long prompt:[/yellow] Encountered an error")
        console.print(f"    - Error: {long_result['error']}")
        console.print()

    if exceed_result["error"]:
        console.print("  • [red]Exceeding limit:[/red] Model rejects prompts that exceed context window or rate limits")
        console.print(f"    - Estimated: {exceed_result['estimated_tokens']} tokens")
        console.print(f"    - Context limit: {context_limit:,} tokens")
        if exceed_result["estimated_tokens"] > context_limit:
            console.print(f"    - Prompt exceeds context limit by: {exceed_result['estimated_tokens'] - context_limit:,} tokens")
        console.print(f"    - Error: {exceed_result['error']}")
        console.print()
    elif exceed_result["usage"]:
        console.print("  • [yellow]Exceeding limit:[/yellow] Note: Prompt may not have exceeded limit")
        console.print(f"    - Actual tokens: {exceed_result['usage']['total_tokens']} tokens")
        console.print()

    console.print("[bold]Token Counting Insights:[/bold]")
    console.print("  • tiktoken provides accurate pre-request token estimation")
    console.print("  • API response includes exact token counts in the 'usage' field")
    console.print("  • Message formatting adds overhead (~3 tokens per message)")
    console.print("  • Models have hard context limits that cannot be exceeded")
    console.print("  • Rate limits (TPM) may be hit before context limits for very large requests")
    console.print()


def run() -> None:
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model
    context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)

    console.print("[bold cyan]Day 5 — Working with Tokens[/bold cyan]")
    console.print()
    console.print("[dim]This demonstration shows token counting for requests and responses,[/dim]")
    console.print("[dim]comparing three cases: short prompt, long prompt, and prompt exceeding context limit.[/dim]")
    console.print()
    console.print(f"[dim]Model:[/dim] {model}  [dim]Base URL:[/dim] {base_url}")
    console.print(f"[dim]Context Limit:[/dim] {context_limit:,} tokens")
    console.print()

    results = []

    for test_case in TEST_CASES:
        console.print(f"[yellow]Running: {test_case['name']}...[/yellow]")

        if test_case["name"] == "Short Prompt":
            prompt = test_case["prompt"]
        elif test_case["name"] == "Long Prompt":
            prompt = create_long_prompt(test_case["target_tokens"], model)
        else:
            exceed_target = int(context_limit * 1.1)
            prompt = create_long_prompt(exceed_target, model)

        result = run_prompt_with_token_counting(prompt, model, base_url, api_key)
        results.append(result)

        display_token_comparison(test_case, result, model, context_limit)

    analyze_results(results, model, context_limit)

