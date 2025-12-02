import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from day13_document_indexing.task import DocumentIndex
from day14_rag_query.task import (
    _build_headers,
    _build_payload,
    _call_llm,
    _extract_message_content,
    _get_base_url,
    _post_chat_completions,
    query_with_rag,
)
from utils.config import settings
from utils.helpers import console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.columns import Columns


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
) -> List[Dict[str, Any]]:
    """Rerank chunks using cross-encoder (LLM-based relevance scoring)."""
    if not chunks:
        return []

    console.print(f"[dim]Reranking {len(chunks)} chunks using cross-encoder...[/dim]")

    reranked_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.get("chunk_text", "")

        system_prompt = "You are a relevance scorer. Rate how relevant a text chunk is to a query on a scale from 0.0 to 1.0, where 1.0 is highly relevant and 0.0 is not relevant at all. Respond with only a single floating-point number between 0.0 and 1.0."

        user_prompt = f"""Query: {query}

Chunk:
{chunk_text}

Rate the relevance of this chunk to the query. Respond with only a number between 0.0 and 1.0."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = _call_llm(base_url, api_key, model, messages, temperature=0.0, max_tokens=10)

        relevance_score = 0.0
        if response:
            numbers = re.findall(r"0?\.\d+|1\.0|0\.0|\d+\.\d+", response)
            if numbers:
                try:
                    relevance_score = float(numbers[0])
                    relevance_score = max(0.0, min(1.0, relevance_score))
                except ValueError:
                    relevance_score = 0.0

        reranked_chunk = chunk.copy()
        reranked_chunk["relevance_score"] = relevance_score
        reranked_chunks.append(reranked_chunk)

    reranked_chunks.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

    for i, chunk in enumerate(reranked_chunks):
        chunk["rerank"] = i + 1

    return reranked_chunks


def filter_by_threshold(
    chunks: List[Dict[str, Any]],
    threshold: float,
    threshold_type: str = "relevance_score",
) -> List[Dict[str, Any]]:
    """Filter chunks by similarity threshold."""
    if threshold_type == "relevance_score":
        filtered = [chunk for chunk in chunks if chunk.get("relevance_score", 0.0) >= threshold]
    elif threshold_type == "distance":
        filtered = [chunk for chunk in chunks if chunk.get("distance", float("inf")) <= threshold]
    else:
        filtered = chunks

    return filtered


def query_with_rag_reranked(
    question: str,
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    model: str,
    k: int = 5,
    temperature: Optional[float] = None,
    rerank: bool = True,
    threshold: Optional[float] = None,
    threshold_type: str = "relevance_score",
) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    """Query with RAG: retrieve, rerank, filter, then send to LLM."""
    console.print("[dim]Searching for relevant chunks...[/dim]")
    chunks = index.search(question, api_key, base_url, k=k)

    if not chunks:
        console.print("[yellow]No relevant chunks found. Proceeding without context.[/yellow]")
        return None, [], {"original_count": 0, "after_rerank": 0, "after_filter": 0}

    original_count = len(chunks)
    stats = {"original_count": original_count, "after_rerank": original_count, "after_filter": original_count}

    if rerank:
        chunks = rerank_chunks(question, chunks, api_key, base_url, model)
        stats["after_rerank"] = len(chunks)

    if threshold is not None:
        chunks = filter_by_threshold(chunks, threshold, threshold_type)
        stats["after_filter"] = len(chunks)

    if not chunks:
        console.print("[yellow]All chunks filtered out. Proceeding without context.[/yellow]")
        return None, [], stats

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Chunk {i}]\n{chunk['chunk_text']}\n")

    context = "\n".join(context_parts)

    system_prompt = "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain relevant information, say so."
    user_prompt = f"""Based on the following context, please answer the question. If the context doesn't contain enough information to answer the question, please indicate that.

Context:
{context}

Question: {question}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    console.print("[dim]Sending query to LLM with filtered RAG context...[/dim]")
    answer = _call_llm(base_url, api_key, model, messages, temperature)

    return answer, chunks, stats


def compare_rag_with_filtering(
    question: str,
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    model: str,
    k: int = 5,
    temperature: Optional[float] = None,
    threshold: float = 0.5,
) -> None:
    """Compare RAG answers with and without filtering/reranking."""
    console.print()
    console.print(f"[bold cyan]Question:[/bold cyan] {question}")
    console.print()

    console.print("[bold]Step 1: RAG without filtering/reranking[/bold]")
    rag_answer, original_chunks = query_with_rag(question, index, api_key, base_url, model, k, temperature)
    console.print()

    console.print("[bold]Step 2: RAG with reranking and filtering[/bold]")
    filtered_answer, filtered_chunks, stats = query_with_rag_reranked(
        question, index, api_key, base_url, model, k, temperature, rerank=True, threshold=threshold
    )
    console.print()

    console.print("[bold]Comparison Results:[/bold]")
    console.print()

    stats_table = Table(show_header=True, header_style="bold magenta", title="Filtering Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Original chunks retrieved", str(stats["original_count"]))
    stats_table.add_row("After reranking", str(stats["after_rerank"]))
    stats_table.add_row("After filtering", str(stats["after_filter"]))
    stats_table.add_row("Chunks filtered out", str(stats["original_count"] - stats["after_filter"]))

    if filtered_chunks:
        avg_relevance = sum(chunk.get("relevance_score", 0.0) for chunk in filtered_chunks) / len(filtered_chunks)
        stats_table.add_row("Average relevance score", f"{avg_relevance:.3f}")
        stats_table.add_row("Threshold used", f"{threshold:.2f}")

    console.print(stats_table)
    console.print()

    if original_chunks:
        console.print("[bold]Original Chunks (before filtering):[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Document", style="white")
        table.add_column("Chunk", style="dim", width=8)
        table.add_column("Distance", style="yellow", width=10)
        table.add_column("Preview", style="dim", width=50)

        for chunk in original_chunks:
            doc_path = Path(chunk["doc_id"])
            preview = chunk["chunk_text"][:100].replace("\n", " ")
            if len(chunk["chunk_text"]) > 100:
                preview += "..."

            table.add_row(
                str(chunk["rank"]),
                doc_path.name,
                f"{chunk['chunk_index'] + 1}/{chunk['total_chunks']}",
                f"{chunk['distance']:.4f}",
                preview,
            )

        console.print(table)
        console.print()

    if filtered_chunks:
        console.print("[bold]Filtered Chunks (after reranking & filtering):[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rerank", style="cyan", width=6)
        table.add_column("Document", style="white")
        table.add_column("Chunk", style="dim", width=8)
        table.add_column("Relevance", style="green", width=10)
        table.add_column("Preview", style="dim", width=50)

        for chunk in filtered_chunks:
            doc_path = Path(chunk["doc_id"])
            preview = chunk["chunk_text"][:100].replace("\n", " ")
            if len(chunk["chunk_text"]) > 100:
                preview += "..."

            relevance = chunk.get("relevance_score", 0.0)
            relevance_str = f"{relevance:.3f}"

            table.add_row(
                str(chunk.get("rerank", "-")),
                doc_path.name,
                f"{chunk['chunk_index'] + 1}/{chunk['total_chunks']}",
                relevance_str,
                preview,
            )

        console.print(table)
        console.print()

    panels = []
    if rag_answer:
        rag_panel = Panel(
            Markdown(rag_answer, code_theme="monokai"),
            title="[bold yellow]RAG without Filtering[/bold yellow]",
            border_style="yellow",
        )
        panels.append(rag_panel)
    else:
        panels.append(
            Panel(
                "[red]Failed to get RAG answer[/red]",
                title="[bold red]RAG without Filtering[/bold red]",
                border_style="red",
            )
        )

    if filtered_answer:
        filtered_panel = Panel(
            Markdown(filtered_answer, code_theme="monokai"),
            title="[bold green]RAG with Reranking & Filtering[/bold green]",
            border_style="green",
        )
        panels.append(filtered_panel)
    else:
        panels.append(
            Panel(
                "[red]Failed to get filtered RAG answer[/red]",
                title="[bold red]RAG with Reranking & Filtering[/bold red]",
                border_style="red",
            )
        )

    console.print(Columns(panels, equal=True, expand=True))
    console.print()


def tune_threshold(
    question: str,
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    model: str,
    k: int = 5,
) -> None:
    """Interactive threshold tuning to find optimal cutoff."""
    console.print()
    console.print(f"[bold cyan]Question:[/bold cyan] {question}")
    console.print()

    console.print("[dim]Retrieving and reranking chunks...[/dim]")
    chunks = index.search(question, api_key, base_url, k=k)

    if not chunks:
        console.print("[yellow]No chunks found.[/yellow]")
        return

    chunks = rerank_chunks(question, chunks, api_key, base_url, model)
    console.print()

    console.print(f"[green]Retrieved and reranked {len(chunks)} chunks[/green]")
    console.print()

    if chunks:
        scores = [chunk.get("relevance_score", 0.0) for chunk in chunks]
        min_score = min(scores)
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        console.print(f"[bold]Relevance Score Statistics:[/bold]")
        console.print(f"  Min: {min_score:.3f}")
        console.print(f"  Max: {max_score:.3f}")
        console.print(f"  Average: {avg_score:.3f}")
        console.print()

    current_threshold = 0.5

    while True:
        console.print()
        console.print(f"[bold]Current threshold:[/bold] {current_threshold:.2f}")

        filtered = filter_by_threshold(chunks, current_threshold, "relevance_score")
        console.print(f"[green]Chunks passing threshold:[/green] {len(filtered)}/{len(chunks)}")
        console.print()

        if filtered:
            console.print("[bold]Chunks that would be used:[/bold]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", style="cyan", width=6)
            table.add_column("Relevance", style="green", width=10)
            table.add_column("Document", style="white")
            table.add_column("Preview", style="dim", width=60)

            for chunk in filtered[:5]:
                doc_path = Path(chunk["doc_id"])
                preview = chunk["chunk_text"][:80].replace("\n", " ")
                if len(chunk["chunk_text"]) > 80:
                    preview += "..."

                table.add_row(
                    str(chunk.get("rerank", "-")),
                    f"{chunk.get('relevance_score', 0.0):.3f}",
                    doc_path.name,
                    preview,
                )

            console.print(table)
        else:
            console.print("[yellow]No chunks pass the threshold.[/yellow]")

        console.print()
        console.print("[bold]Options:[/bold]")
        console.print("  1. Adjust threshold")
        console.print("  2. Test this threshold with a query")
        console.print("  3. Exit threshold tuning")
        console.print()

        choice = Prompt.ask("Select an option", default="3")

        if choice == "1":
            threshold_str = Prompt.ask(
                "Enter new threshold (0.0-1.0)",
                default=str(current_threshold),
            )
            try:
                new_threshold = float(threshold_str)
                current_threshold = max(0.0, min(1.0, new_threshold))
            except ValueError:
                console.print("[yellow]Invalid number. Using current threshold.[/yellow]")

        elif choice == "2":
            if not filtered:
                console.print("[yellow]No chunks to test with. Adjust threshold first.[/yellow]")
                continue

            console.print()
            console.print("[dim]Testing with current threshold...[/dim]")
            answer, _, stats = query_with_rag_reranked(
                question, index, api_key, base_url, model, k, rerank=True, threshold=current_threshold
            )

            if answer:
                console.print()
                console.print("[bold]Answer with filtered chunks:[/bold]")
                console.print(Panel(Markdown(answer, code_theme="monokai"), border_style="green"))
                console.print()
                console.print(f"[dim]Used {stats['after_filter']} chunks (filtered from {stats['original_count']})[/dim]")

        elif choice == "3":
            break

        else:
            console.print("[yellow]Invalid option. Please try again.[/yellow]")


def run() -> None:
    """Main entry point for Day 15 task."""
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 15 — Reranking & Filtering[/bold cyan]")
    console.print()
    console.print("[dim]This tool improves RAG by reranking and filtering chunks for better relevance.[/dim]")
    console.print()

    index_dir = Path("outputs")
    index_path = index_dir / "document_index.faiss"

    if not index_path.exists():
        console.print("[red]Document index not found. Please run Day 13 first to create an index.[/red]")
        console.print(f"[dim]Expected index at: {index_path}[/dim]")
        return

    index = DocumentIndex(index_path)
    if not index._initialize_faiss():
        console.print("[red]Failed to initialize FAISS index.[/red]")
        return

    stats = index.get_stats()
    if stats["total_vectors"] == 0:
        console.print("[yellow]Index is empty. Please index some documents first using Day 13.[/yellow]")
        return

    console.print(f"[green]Loaded index with {stats['total_vectors']} vectors from {stats['documents']} documents[/green]")
    console.print()

    while True:
        console.print()
        console.print("[bold]Options:[/bold]")
        console.print("  1. Compare RAG with/without filtering")
        console.print("  2. Tune threshold interactively")
        console.print("  3. Exit")
        console.print()

        choice = Prompt.ask("Select an option", default="3")

        if choice == "1":
            question = Prompt.ask("Enter your question")
            if not question:
                continue

            k = int(Prompt.ask("Number of chunks to retrieve", default="5"))
            threshold_str = Prompt.ask("Relevance threshold (0.0-1.0)", default="0.5")
            try:
                threshold = float(threshold_str)
                threshold = max(0.0, min(1.0, threshold))
            except ValueError:
                console.print("[yellow]Invalid threshold. Using default 0.5.[/yellow]")
                threshold = 0.5

            compare_rag_with_filtering(question, index, api_key, base_url, model, k=k, threshold=threshold)

        elif choice == "2":
            question = Prompt.ask("Enter your question")
            if not question:
                continue

            k = int(Prompt.ask("Number of chunks to retrieve", default="5"))

            tune_threshold(question, index, api_key, base_url, model, k=k)

        elif choice == "3":
            console.print("[yellow]Goodbye![/yellow]")
            break

        else:
            console.print("[yellow]Invalid option. Please try again.[/yellow]")

