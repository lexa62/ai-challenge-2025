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
from day15_reranking_filtering.task import (
    filter_by_threshold,
    query_with_rag_reranked,
    rerank_chunks,
)
from utils.config import settings
from utils.helpers import console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.columns import Columns


def query_with_rag_citations(
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
    """Query with RAG: retrieve, rerank, filter, then send to LLM with citation enforcement."""
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
    chunk_citations = {}
    for i, chunk in enumerate(chunks, 1):
        citation_id = f"[{i}]"
        chunk_citations[citation_id] = {
            "chunk_index": i - 1,
            "doc_id": chunk.get("doc_id", ""),
            "chunk_text": chunk.get("chunk_text", ""),
        }
        chunk["citation_id"] = citation_id
        context_parts.append(f"{citation_id} {chunk['chunk_text']}\n")

    context = "\n".join(context_parts)

    system_prompt = """You are a helpful assistant that answers questions based on the provided context.
You MUST cite your sources using the citation format [1], [2], etc. for every factual claim you make.
Each numbered citation corresponds to a source chunk provided in the context.
If you make multiple claims, cite the relevant source(s) for each claim.
If the context doesn't contain relevant information, say so explicitly without citations."""

    user_prompt = f"""Based on the following context, please answer the question.
You MUST include citations [1], [2], etc. for every factual claim, referencing the source chunks below.

Context:
{context}

Question: {question}

Remember: You must cite sources using [1], [2], etc. for every claim you make based on the context."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    console.print("[dim]Sending query to LLM with citation-enforced RAG context...[/dim]")
    answer = _call_llm(base_url, api_key, model, messages, temperature)

    stats["chunk_citations"] = chunk_citations
    return answer, chunks, stats


def validate_citations(
    answer: str,
    chunks: List[Dict[str, Any]],
    chunk_citations: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Parse and validate citations in the answer."""
    if not answer:
        return {
            "has_citations": False,
            "valid_citations": [],
            "invalid_citations": [],
            "missing_citations": [],
            "citation_count": 0,
        }

    citation_pattern = r"\[(\d+)\]"
    found_citations = re.findall(citation_pattern, answer)
    found_citation_ids = [f"[{num}]" for num in found_citations]

    valid_citations = []
    invalid_citations = []
    valid_citation_nums = set()

    for citation_id in found_citation_ids:
        if citation_id in chunk_citations:
            valid_citations.append(citation_id)
            valid_citation_nums.add(int(citation_id.strip("[]")))
        else:
            invalid_citations.append(citation_id)

    all_chunk_nums = set(range(1, len(chunks) + 1))
    missing_citations = [f"[{num}]" for num in all_chunk_nums if num not in valid_citation_nums]

    return {
        "has_citations": len(found_citations) > 0,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "missing_citations": missing_citations,
        "citation_count": len(found_citations),
        "valid_count": len(valid_citations),
        "invalid_count": len(invalid_citations),
    }


def compare_with_without_citations(
    question: str,
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    model: str,
    k: int = 5,
    temperature: Optional[float] = None,
    threshold: float = 0.5,
) -> None:
    """Compare RAG answers with and without citation enforcement."""
    console.print()
    console.print(f"[bold cyan]Question:[/bold cyan] {question}")
    console.print()

    console.print("[bold]Step 1: RAG without citation enforcement[/bold]")
    answer_no_citations, chunks_no_citations = query_with_rag(question, index, api_key, base_url, model, k, temperature)
    console.print()

    console.print("[bold]Step 2: RAG with citation enforcement[/bold]")
    answer_with_citations, chunks_with_citations, stats = query_with_rag_citations(
        question, index, api_key, base_url, model, k, temperature, rerank=True, threshold=threshold
    )
    console.print()

    validation = {}
    if answer_with_citations:
        validation = validate_citations(
            answer_with_citations, chunks_with_citations, stats.get("chunk_citations", {})
        )

    console.print("[bold]Comparison Results:[/bold]")
    console.print()

    if chunks_with_citations:
        console.print("[bold]Retrieved Chunks (with citation IDs):[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Citation", style="cyan", width=10)
        table.add_column("Document", style="white")
        table.add_column("Chunk", style="dim", width=8)
        table.add_column("Relevance", style="green", width=10)
        table.add_column("Preview", style="dim", width=50)

        for chunk in chunks_with_citations:
            doc_path = Path(chunk.get("doc_id", ""))
            preview = chunk.get("chunk_text", "")[:100].replace("\n", " ")
            if len(chunk.get("chunk_text", "")) > 100:
                preview += "..."

            citation_id = chunk.get("citation_id", "-")
            relevance = chunk.get("relevance_score", 0.0)
            relevance_str = f"{relevance:.3f}" if relevance > 0 else "-"

            table.add_row(
                citation_id,
                doc_path.name if doc_path.name else str(doc_path),
                f"{chunk.get('chunk_index', 0) + 1}/{chunk.get('total_chunks', 0)}",
                relevance_str,
                preview,
            )

        console.print(table)
        console.print()

    if validation:
        validation_table = Table(show_header=True, header_style="bold magenta", title="Citation Validation")
        validation_table.add_column("Metric", style="cyan")
        validation_table.add_column("Value", style="white")

        validation_table.add_row("Has citations", "Yes" if validation["has_citations"] else "No")
        validation_table.add_row("Total citations found", str(validation["citation_count"]))
        validation_table.add_row("Valid citations", str(validation["valid_count"]))
        validation_table.add_row("Invalid citations", str(validation["invalid_count"]))
        validation_table.add_row("Missing chunk citations", str(len(validation["missing_citations"])))

        if validation["invalid_citations"]:
            validation_table.add_row("Invalid citation IDs", ", ".join(validation["invalid_citations"]))

        if validation["missing_citations"]:
            validation_table.add_row("Uncited chunks", ", ".join(validation["missing_citations"]))

        console.print(validation_table)
        console.print()

    panels = []
    if answer_no_citations:
        no_citations_panel = Panel(
            Markdown(answer_no_citations, code_theme="monokai"),
            title="[bold yellow]Without Citation Enforcement[/bold yellow]",
            border_style="yellow",
        )
        panels.append(no_citations_panel)
    else:
        panels.append(
            Panel(
                "[red]Failed to get answer[/red]",
                title="[bold red]Without Citation Enforcement[/bold red]",
                border_style="red",
            )
        )

    if answer_with_citations:
        with_citations_panel = Panel(
            Markdown(answer_with_citations, code_theme="monokai"),
            title="[bold green]With Citation Enforcement[/bold green]",
            border_style="green",
        )
        panels.append(with_citations_panel)
    else:
        panels.append(
            Panel(
                "[red]Failed to get answer[/red]",
                title="[bold red]With Citation Enforcement[/bold red]",
                border_style="red",
            )
        )

    console.print(Columns(panels, equal=True, expand=True))
    console.print()


def test_citations(
    questions: List[str],
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    model: str,
    k: int = 5,
    temperature: Optional[float] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Test citations on multiple questions and return results."""
    results = []

    for i, question in enumerate(questions, 1):
        console.print()
        console.print(f"[bold cyan]Test {i}/{len(questions)}:[/bold cyan] {question}")
        console.print()

        answer, chunks, stats = query_with_rag_citations(
            question, index, api_key, base_url, model, k, temperature, rerank=True, threshold=threshold
        )

        validation = {}
        if answer:
            validation = validate_citations(answer, chunks, stats.get("chunk_citations", {}))

        results.append(
            {
                "question": question,
                "answer": answer,
                "chunks": chunks,
                "validation": validation,
                "stats": stats,
            }
        )

        if validation:
            has_citations = validation.get("has_citations", False)
            valid_count = validation.get("valid_count", 0)
            status = "✓" if has_citations and valid_count > 0 else "✗"
            console.print(f"[{'green' if has_citations else 'red'}]{status} Citations: {valid_count} valid, {validation.get('invalid_count', 0)} invalid[/]")

    return results


def display_test_results(results: List[Dict[str, Any]]) -> None:
    """Display comprehensive test results."""
    console.print()
    console.print("[bold cyan]Test Results Summary[/bold cyan]")
    console.print()

    summary_table = Table(show_header=True, header_style="bold magenta", title="Citation Test Summary")
    summary_table.add_column("Question #", style="cyan", width=10)
    summary_table.add_column("Has Citations", style="white", width=12)
    summary_table.add_column("Valid", style="green", width=8)
    summary_table.add_column("Invalid", style="red", width=8)
    summary_table.add_column("Missing", style="yellow", width=8)
    summary_table.add_column("Status", style="white", width=10)

    total_has_citations = 0
    total_valid = 0
    total_invalid = 0

    for i, result in enumerate(results, 1):
        validation = result.get("validation", {})
        has_citations = validation.get("has_citations", False)
        valid_count = validation.get("valid_count", 0)
        invalid_count = validation.get("invalid_count", 0)
        missing_count = len(validation.get("missing_citations", []))

        if has_citations:
            total_has_citations += 1
        total_valid += valid_count
        total_invalid += invalid_count

        status = "✓ Pass" if has_citations and valid_count > 0 else "✗ Fail"
        status_style = "green" if has_citations and valid_count > 0 else "red"

        summary_table.add_row(
            str(i),
            "Yes" if has_citations else "No",
            str(valid_count),
            str(invalid_count),
            str(missing_count),
            f"[{status_style}]{status}[/]",
        )

    console.print(summary_table)
    console.print()

    console.print(f"[bold]Overall Statistics:[/bold]")
    console.print(f"  Questions with citations: {total_has_citations}/{len(results)}")
    console.print(f"  Total valid citations: {total_valid}")
    console.print(f"  Total invalid citations: {total_invalid}")
    console.print(f"  Citation rate: {(total_has_citations / len(results) * 100):.1f}%")
    console.print()

    for i, result in enumerate(results, 1):
        console.print()
        console.print(f"[bold]Question {i}:[/bold] {result['question']}")
        console.print()

        if result.get("answer"):
            validation = result.get("validation", {})
            console.print(f"[dim]Citations found: {validation.get('citation_count', 0)}[/dim]")
            console.print(f"[dim]Valid: {validation.get('valid_count', 0)}, Invalid: {validation.get('invalid_count', 0)}[/dim]")

            answer_panel = Panel(
                Markdown(result["answer"], code_theme="monokai"),
                title=f"[bold]Answer {i}[/bold]",
                border_style="cyan",
            )
            console.print(answer_panel)
        else:
            console.print("[red]No answer generated[/red]")


def run() -> None:
    """Main entry point for Day 16 task."""
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 16 — Citations & Sources[/bold cyan]")
    console.print()
    console.print("[dim]This tool enhances RAG to enforce citations in every answer.[/dim]")
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
        console.print("  1. Test citations on a question")
        console.print("  2. Compare with/without citations")
        console.print("  3. Run batch test (4-5 questions)")
        console.print("  4. Exit")
        console.print()

        choice = Prompt.ask("Select an option", default="4")

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

            answer, chunks, stats_result = query_with_rag_citations(
                question, index, api_key, base_url, model, k=k, rerank=True, threshold=threshold
            )

            if answer:
                validation = validate_citations(answer, chunks, stats_result.get("chunk_citations", {}))

                console.print()
                console.print("[bold]Answer with Citations:[/bold]")
                console.print(Panel(Markdown(answer, code_theme="monokai"), border_style="green"))
                console.print()

                if validation:
                    console.print("[bold]Citation Validation:[/bold]")
                    validation_table = Table(show_header=True, header_style="bold magenta")
                    validation_table.add_column("Metric", style="cyan")
                    validation_table.add_column("Value", style="white")

                    validation_table.add_row("Has citations", "Yes" if validation["has_citations"] else "[red]No[/red]")
                    validation_table.add_row("Total citations", str(validation["citation_count"]))
                    validation_table.add_row("Valid citations", f"[green]{validation['valid_count']}[/green]")
                    validation_table.add_row("Invalid citations", f"[red]{validation['invalid_count']}[/red]")

                    if validation["invalid_citations"]:
                        validation_table.add_row("Invalid IDs", ", ".join(validation["invalid_citations"]))

                    if validation["missing_citations"]:
                        validation_table.add_row("Uncited chunks", ", ".join(validation["missing_citations"]))

                    console.print(validation_table)

        elif choice == "2":
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

            compare_with_without_citations(question, index, api_key, base_url, model, k=k, threshold=threshold)

        elif choice == "3":
            console.print()
            console.print("[bold]Batch Testing with 4-5 Questions[/bold]")
            console.print()

            questions = []
            for i in range(5):
                q = Prompt.ask(f"Enter question {i + 1} (or press Enter to finish)", default="")
                if not q:
                    break
                questions.append(q)

            if not questions:
                console.print("[yellow]No questions provided.[/yellow]")
                continue

            k = int(Prompt.ask("Number of chunks to retrieve", default="5"))
            threshold_str = Prompt.ask("Relevance threshold (0.0-1.0)", default="0.5")
            try:
                threshold = float(threshold_str)
                threshold = max(0.0, min(1.0, threshold))
            except ValueError:
                console.print("[yellow]Invalid threshold. Using default 0.5.[/yellow]")
                threshold = 0.5

            results = test_citations(questions, index, api_key, base_url, model, k=k, threshold=threshold)
            display_test_results(results)

        elif choice == "4":
            console.print("[yellow]Goodbye![/yellow]")
            break

        else:
            console.print("[yellow]Invalid option. Please try again.[/yellow]")


