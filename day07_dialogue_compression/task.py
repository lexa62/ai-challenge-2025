import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests
import tiktoken
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from utils.config import settings

console = Console()

MODEL_ENCODINGS = {
    "gpt-4o-mini": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4-turbo": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


@dataclass
class CompressionEvent:
    message_count: int
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    messages_compressed: int


@dataclass
class TokenStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    compression_events: List[CompressionEvent] = field(default_factory=list)
    cumulative_tokens: List[int] = field(default_factory=list)


def get_model_encoding(model: str) -> str:
    encoding_name = MODEL_ENCODINGS.get(model, "cl100k_base")
    try:
        tiktoken.get_encoding(encoding_name)
        return encoding_name
    except Exception:
        return "cl100k_base"


def count_tokens(text: str, model: str) -> int:
    encoding_name = get_model_encoding(model)
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        return len(text.split())


def count_messages_tokens(messages: List[Dict[str, str]], model: str) -> int:
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
    except Exception:
        total_text = " ".join([str(msg.get("content", "")) for msg in messages])
        return len(total_text.split())


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com")


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


def _extract_message_content(body: Dict[str, Any]) -> str:
    try:
        return (body["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def _extract_usage(body: Dict[str, Any]) -> Optional[Dict[str, int]]:
    try:
        usage = body.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    except Exception:
        return None


def compress_messages(
    messages_to_compress: List[Dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
) -> Optional[str]:
    if not messages_to_compress:
        return None

    summary_prompt = (
        "Please provide a concise summary of the following conversation history. "
        "Preserve all important information, key decisions, user preferences, and context "
        "that would be needed to continue the conversation naturally. "
        "Focus on facts, preferences, and decisions rather than exact wording.\n\n"
        "Conversation history:\n"
    )

    conversation_text = ""
    for msg in messages_to_compress:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"User: {content}\n\n"
        elif role == "assistant":
            conversation_text += f"Assistant: {content}\n\n"

    full_prompt = summary_prompt + conversation_text

    summary_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that creates concise summaries of conversations while preserving all important information.",
        },
        {"role": "user", "content": full_prompt},
    ]

    payload = {
        "model": model,
        "messages": summary_messages,
        "temperature": 0.3,
    }

    try:
        body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=60.0)
        summary = _extract_message_content(body)
        return summary
    except Exception as e:
        console.print(f"[yellow]Warning: Compression failed: {e}[/yellow]")
        return None


def should_compress(
    messages: List[Dict[str, str]], compression_threshold: int = 10, keep_recent: int = 2
) -> Tuple[bool, int]:
    user_assistant_pairs = 0
    system_count = 0

    for msg in messages:
        if msg.get("role") == "system":
            system_count += 1
        elif msg.get("role") in ("user", "assistant"):
            if msg.get("role") == "user":
                user_assistant_pairs += 1

    if user_assistant_pairs < compression_threshold + keep_recent:
        return False, 0

    messages_to_compress = user_assistant_pairs - keep_recent
    return True, messages_to_compress


def count_user_assistant_pairs(messages: List[Dict[str, str]]) -> int:
    count = 0
    for msg in messages:
        if msg.get("role") == "user":
            count += 1
    return count


def run_conversation_with_compression(
    conversation_script: List[str],
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    compression_threshold: int = 10,
    keep_recent: int = 2,
    track_stats: bool = True,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, str]], TokenStats, List[str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    stats = TokenStats()
    responses: List[str] = []
    total_questions = len(conversation_script)

    for idx, user_message in enumerate(conversation_script, 1):
        if show_progress:
            console.print(f"[cyan]Question {idx}/{total_questions}:[/cyan] {user_message}")
        pairs_before = count_user_assistant_pairs(messages)

        if pairs_before >= compression_threshold + keep_recent:
            pairs_to_compress = pairs_before - keep_recent

            system_msg = messages[0] if messages and messages[0].get("role") == "system" else None

            messages_to_compress = []
            recent_messages = []

            if system_msg:
                non_system_messages = messages[1:]
            else:
                non_system_messages = messages

            if keep_recent > 0:
                recent_count = keep_recent * 2
                if len(non_system_messages) > recent_count:
                    messages_to_compress = non_system_messages[:-recent_count]
                    recent_messages = non_system_messages[-recent_count:]
                else:
                    messages_to_compress = []
                    recent_messages = non_system_messages
            else:
                messages_to_compress = non_system_messages
                recent_messages = []

            if messages_to_compress:
                tokens_before = count_messages_tokens(messages, model)

                console.print(f"[dim][Compression] Compressing {pairs_to_compress} message pairs...[/dim]")
                summary = compress_messages(messages_to_compress, base_url, api_key, model)

                if summary:
                    summary_preview = summary[:300] + "..." if len(summary) > 300 else summary
                    console.print(f"[blue][Compression Summary][/blue]")
                    console.print(Panel(summary_preview, border_style="blue", title="Compressed History"))

                    new_messages: List[Dict[str, str]] = []
                    if system_msg:
                        new_messages.append(system_msg)
                    new_messages.append(
                        {
                            "role": "system",
                            "content": f"Previous conversation summary: {summary}",
                        }
                    )
                    new_messages.extend(recent_messages)
                    messages = new_messages

                    tokens_after = count_messages_tokens(messages, model)
                    tokens_saved = tokens_before - tokens_after

                    if track_stats:
                        stats.compression_events.append(
                            CompressionEvent(
                                message_count=pairs_before,
                                tokens_before=tokens_before,
                                tokens_after=tokens_after,
                                tokens_saved=tokens_saved,
                                messages_compressed=pairs_to_compress,
                            )
                        )
                    console.print(f"[dim][Compression] Saved ~{tokens_saved} tokens[/dim]")
                    console.print()

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        }

        try:
            if show_progress:
                console.print("[dim]Waiting for response...[/dim]")
            body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=60.0)
            assistant_response = _extract_message_content(body)
            usage = _extract_usage(body)

            if assistant_response:
                messages.append({"role": "assistant", "content": assistant_response})
                responses.append(assistant_response)

                if show_progress:
                    response_preview = assistant_response[:200] + "..." if len(assistant_response) > 200 else assistant_response
                    console.print(f"[green]Response:[/green] {response_preview}")
                    if usage:
                        console.print(f"[dim]Tokens: {usage['total_tokens']:,} (prompt: {usage['prompt_tokens']:,}, completion: {usage['completion_tokens']:,})[/dim]")
                    console.print()

                if track_stats and usage:
                    stats.total_prompt_tokens += usage["prompt_tokens"]
                    stats.total_completion_tokens += usage["completion_tokens"]
                    stats.total_tokens += usage["total_tokens"]
                    stats.cumulative_tokens.append(stats.total_tokens)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            responses.append("")

    return messages, stats, responses


def run_conversation_without_compression(
    conversation_script: List[str],
    system_prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    track_stats: bool = True,
    show_progress: bool = True,
) -> Tuple[List[Dict[str, str]], TokenStats, List[str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    stats = TokenStats()
    responses: List[str] = []
    total_questions = len(conversation_script)

    for idx, user_message in enumerate(conversation_script, 1):
        if show_progress:
            console.print(f"[cyan]Question {idx}/{total_questions}:[/cyan] {user_message}")

        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
        }

        try:
            if show_progress:
                console.print("[dim]Waiting for response...[/dim]")
            body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=60.0)
            assistant_response = _extract_message_content(body)
            usage = _extract_usage(body)

            if assistant_response:
                messages.append({"role": "assistant", "content": assistant_response})
                responses.append(assistant_response)

                if show_progress:
                    response_preview = assistant_response[:200] + "..." if len(assistant_response) > 200 else assistant_response
                    console.print(f"[green]Response:[/green] {response_preview}")
                    if usage:
                        console.print(f"[dim]Tokens: {usage['total_tokens']:,} (prompt: {usage['prompt_tokens']:,}, completion: {usage['completion_tokens']:,})[/dim]")
                    console.print()

                if track_stats and usage:
                    stats.total_prompt_tokens += usage["prompt_tokens"]
                    stats.total_completion_tokens += usage["completion_tokens"]
                    stats.total_tokens += usage["total_tokens"]
                    stats.cumulative_tokens.append(stats.total_tokens)

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            responses.append("")

    return messages, stats, responses


def create_research_scenario() -> Tuple[str, List[str]]:
    system_prompt = (
        "You are a helpful research assistant helping a user learn about quantum computing. "
        "Answer questions clearly and build upon previous information shared in the conversation. "
        "Maintain context about what has been discussed and reference earlier points when relevant."
    )

    conversation_script = [
        "What is quantum computing?",
        "How does it differ from classical computing?",
        "What are qubits?",
        "Can you explain superposition?",
        "What about entanglement?",
        "What are some practical applications?",
        "Which companies are working on quantum computers?",
        "What are the main challenges?",
        "How does quantum error correction work?",
        "What is a quantum algorithm?",
        "Can you give an example?",
        "How long until we have practical quantum computers?",
        "What is quantum supremacy?",
        "Has it been achieved?",
        "What are the security implications?",
    ]

    return system_prompt, conversation_script


def display_conversation(
    messages: List[Dict[str, str]], title: str = "Conversation History"
) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()

    user_count = 0
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "system":
            if "Previous conversation summary" in content:
                console.print("[dim][System - Summary][/dim]")
                summary_preview = content[:400] + "..." if len(content) > 400 else content
                console.print(Panel(summary_preview, border_style="blue", title="Compressed History"))
                console.print()
            else:
                console.print(f"[dim][System Prompt][/dim] {content[:150]}...")
                console.print()
        elif role == "user":
            user_count += 1
            console.print(f"[bold cyan]Q{user_count}:[/bold cyan] {content}")
            console.print()
        elif role == "assistant":
            response_preview = content[:400] + "..." if len(content) > 400 else content
            console.print(f"[bold green]A{user_count}:[/bold green] {response_preview}")
            console.print()

    console.print()


def compare_responses(
    responses_with: List[str],
    responses_without: List[str],
    conversation_script: List[str],
    base_url: str,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    if len(responses_with) != len(responses_without):
        return {
            "length_match": False,
            "quality_score": 0.0,
            "semantic_similarity": 0.0,
            "information_retention": 0.0,
            "notes": ["Response counts don't match"],
        }

    quality_scores = []
    semantic_similarities = []
    information_retentions = []
    notes = []
    total_responses = len(responses_with)

    for i, (resp_with, resp_without, question) in enumerate(zip(responses_with, responses_without, conversation_script), 1):
        console.print(f"[dim]Evaluating response {i}/{total_responses}...[/dim]")
        len_with = len(resp_with)
        len_without = len(resp_without)
        len_ratio = len_with / len_without if len_without > 0 else 0

        quality_score = 1.0
        if len_ratio < 0.3:
            quality_score -= 0.15
            notes.append(f"Response {i}: Compressed version very short ({len_ratio:.1%} of original)")
        elif len_ratio < 0.5:
            quality_score -= 0.08
            notes.append(f"Response {i}: Compressed version shorter ({len_ratio:.1%} of original)")
        elif len_ratio > 1.5:
            quality_score -= 0.05
            notes.append(f"Response {i}: Compressed version longer ({len_ratio:.1%} of original)")

        quality_scores.append(quality_score)

        comparison_prompt = (
            f"Compare these two responses to the same question and rate them:\n\n"
            f"Question: {question}\n\n"
            f"Response A (without compression):\n{resp_without}\n\n"
            f"Response B (with compression):\n{resp_with}\n\n"
            f"Rate on a scale of 0.0 to 1.0:\n"
            f"1. Semantic similarity: How similar are the core meanings? (0.0-1.0)\n"
            f"2. Information retention: Does Response B contain the key information from Response A? (0.0-1.0)\n\n"
            f"Respond ONLY with a JSON object: {{\"semantic_similarity\": 0.0-1.0, \"information_retention\": 0.0-1.0}}"
        )

        try:
            comparison_messages = [
                {
                    "role": "system",
                    "content": "You are an expert evaluator. Respond only with valid JSON.",
                },
                {"role": "user", "content": comparison_prompt},
            ]

            payload = {
                "model": model,
                "messages": comparison_messages,
                "temperature": 0.3,
            }

            body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=30.0)
            comparison_result = _extract_message_content(body)

            try:
                import json
                comparison_json = json.loads(comparison_result)
                semantic_sim = float(comparison_json.get("semantic_similarity", 0.5))
                info_ret = float(comparison_json.get("information_retention", 0.5))
                semantic_similarities.append(semantic_sim)
                information_retentions.append(info_ret)
            except (json.JSONDecodeError, ValueError, KeyError):
                semantic_similarities.append(0.5)
                information_retentions.append(0.5)
                notes.append(f"Response {i}: Could not parse quality metrics")

        except Exception:
            semantic_similarities.append(0.5)
            information_retentions.append(0.5)

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_semantic = sum(semantic_similarities) / len(semantic_similarities) if semantic_similarities else 0.0
    avg_info_ret = sum(information_retentions) / len(information_retentions) if information_retentions else 0.0

    overall_score = (avg_quality * 0.3 + avg_semantic * 0.35 + avg_info_ret * 0.35)

    return {
        "length_match": True,
        "quality_score": overall_score,
        "semantic_similarity": avg_semantic,
        "information_retention": avg_info_ret,
        "notes": notes,
    }


def display_comparison_results(
    stats_with: TokenStats,
    stats_without: TokenStats,
    responses_with: List[str],
    responses_without: List[str],
    model: str,
    base_url: str,
    api_key: str,
    conversation_script: List[str],
) -> None:
    console.print()
    console.rule("[bold cyan]Comparison Results[/bold cyan]")
    console.print()

    comparison_table = Table(show_header=True, header_style="bold magenta")
    comparison_table.add_column("Metric", style="cyan", width=30)
    comparison_table.add_column("Without Compression", style="white", justify="right")
    comparison_table.add_column("With Compression", style="white", justify="right")
    comparison_table.add_column("Difference", style="yellow", justify="right")

    total_without = stats_without.total_tokens
    total_with = stats_with.total_tokens
    token_savings = total_without - total_with
    savings_percent = (token_savings / total_without * 100) if total_without > 0 else 0

    comparison_table.add_row("Total Tokens", f"{total_without:,}", f"{total_with:,}", f"{token_savings:,} ({savings_percent:.1f}%)")
    comparison_table.add_row(
        "Prompt Tokens",
        f"{stats_without.total_prompt_tokens:,}",
        f"{stats_with.total_prompt_tokens:,}",
        f"{stats_without.total_prompt_tokens - stats_with.total_prompt_tokens:,}",
    )
    comparison_table.add_row(
        "Completion Tokens",
        f"{stats_without.total_completion_tokens:,}",
        f"{stats_with.total_completion_tokens:,}",
        f"{stats_without.total_completion_tokens - stats_with.total_completion_tokens:,}",
    )
    comparison_table.add_row(
        "Compression Events",
        "0",
        str(len(stats_with.compression_events)),
        f"+{len(stats_with.compression_events)}",
    )

    console.print(comparison_table)
    console.print()

    if stats_with.compression_events:
        console.print("[bold]Compression Events:[/bold]")
        compression_table = Table(show_header=True, header_style="bold cyan")
        compression_table.add_column("Event", style="cyan", justify="right")
        compression_table.add_column("Pairs Compressed", style="white", justify="right")
        compression_table.add_column("Tokens Before", style="white", justify="right")
        compression_table.add_column("Tokens After", style="white", justify="right")
        compression_table.add_column("Tokens Saved", style="green", justify="right")

        for i, event in enumerate(stats_with.compression_events, 1):
            compression_table.add_row(
                str(i),
                str(event.messages_compressed),
                f"{event.tokens_before:,}",
                f"{event.tokens_after:,}",
                f"{event.tokens_saved:,}",
            )

        console.print(compression_table)
        console.print()

    console.print("[yellow]Evaluating response quality...[/yellow]")
    console.print()
    quality_comparison = compare_responses(
        responses_with, responses_without, conversation_script, base_url, api_key, model
    )

    console.print("[bold]Response Quality Comparison:[/bold]")
    quality_table = Table(show_header=True, header_style="bold cyan")
    quality_table.add_column("Metric", style="cyan")
    quality_table.add_column("Value", style="white")
    quality_table.add_column("Status", style="white")

    quality_table.add_row(
        "Response Count Match",
        "Yes" if quality_comparison["length_match"] else "No",
        "[green]✓[/green]" if quality_comparison["length_match"] else "[red]✗[/red]"
    )

    overall_score = quality_comparison["quality_score"]
    score_status = "[green]Excellent[/green]" if overall_score >= 0.9 else "[yellow]Good[/yellow]" if overall_score >= 0.7 else "[red]Poor[/red]"
    quality_table.add_row("Overall Quality Score", f"{overall_score:.3f}/1.0", score_status)

    semantic_sim = quality_comparison.get("semantic_similarity", 0.0)
    semantic_status = "[green]High[/green]" if semantic_sim >= 0.8 else "[yellow]Medium[/yellow]" if semantic_sim >= 0.6 else "[red]Low[/red]"
    quality_table.add_row("Semantic Similarity", f"{semantic_sim:.3f}/1.0", semantic_status)

    info_ret = quality_comparison.get("information_retention", 0.0)
    info_status = "[green]High[/green]" if info_ret >= 0.8 else "[yellow]Medium[/yellow]" if info_ret >= 0.6 else "[red]Low[/red]"
    quality_table.add_row("Information Retention", f"{info_ret:.3f}/1.0", info_status)

    console.print(quality_table)

    if quality_comparison["notes"]:
        console.print()
        console.print("[bold]Notes:[/bold]")
        for note in quality_comparison["notes"]:
            console.print(f"  • {note}")

    console.print()
    console.print("[bold]Key Observations:[/bold]")
    console.print(f"  • Token savings: {token_savings:,} tokens ({savings_percent:.1f}%)")
    console.print(f"  • Compression events: {len(stats_with.compression_events)}")
    if stats_with.compression_events:
        avg_savings = sum(e.tokens_saved for e in stats_with.compression_events) / len(stats_with.compression_events)
        console.print(f"  • Average savings per compression: {avg_savings:.0f} tokens")
    console.print(f"  • Response quality maintained: {quality_comparison['quality_score'] >= 0.9}")
    console.print()


def run_comparison_test() -> None:
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 7 — Dialogue Compression[/bold cyan]")
    console.print()
    console.print("[dim]This test compares conversations with and without compression.[/dim]")
    console.print("[dim]Compression occurs every 5 user+assistant pairs (10 messages).[/dim]")
    console.print()
    console.print(f"[dim]Model:[/dim] {model}  [dim]Base URL:[/dim] {base_url}")
    console.print()

    system_prompt, conversation_script = create_research_scenario()

    console.print("[yellow]Running conversation WITHOUT compression...[/yellow]")
    console.print()
    messages_without, stats_without, responses_without = run_conversation_without_compression(
        conversation_script, system_prompt, model, base_url, api_key, show_progress=True
    )

    console.print()
    console.print("[yellow]Running conversation WITH compression...[/yellow]")
    console.print()
    messages_with, stats_with, responses_with = run_conversation_with_compression(
        conversation_script, system_prompt, model, base_url, api_key, compression_threshold=5, keep_recent=2, show_progress=True
    )

    console.print()
    display_comparison_results(stats_with, stats_without, responses_with, responses_without, model, base_url, api_key, conversation_script)


def run() -> None:
    run_comparison_test()

