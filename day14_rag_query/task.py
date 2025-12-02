import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from day13_document_indexing.task import DocumentIndex
from utils.config import settings
from utils.helpers import console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.columns import Columns


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
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


def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com")


def _build_payload(
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _call_llm(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    payload = _build_payload(model, messages, temperature, max_tokens)

    try:
        body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=30.0)
        return _extract_message_content(body)
    except requests.Timeout:
        console.print("[red]Request timed out. Try again or adjust your prompt.[/red]")
        return None
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else "unknown"
        snippet = ""
        try:
            err_json = http_err.response.json() if http_err.response is not None else {}
            snippet = err_json.get("error", {}).get("message", "")
        except Exception:
            try:
                snippet = http_err.response.text[:300] if http_err.response is not None else ""
            except Exception:
                snippet = ""
        console.print(f"[red]HTTP {status}[/red] {snippet}")
        return None
    except requests.RequestException as req_err:
        console.print(f"[red]Network error:[/red] {req_err}")
        return None
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON response from server.[/red]")
        return None


def query_with_rag(
    question: str,
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    model: str,
    k: int = 5,
    temperature: Optional[float] = None,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Query with RAG: search for relevant chunks, merge with question, send to LLM."""
    console.print("[dim]Searching for relevant chunks...[/dim]")
    chunks = index.search(question, api_key, base_url, k=k)

    if not chunks:
        console.print("[yellow]No relevant chunks found. Proceeding without context.[/yellow]")
        return query_without_rag(question, api_key, base_url, model, temperature), []

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

    console.print("[dim]Sending query to LLM with RAG context...[/dim]")
    answer = _call_llm(base_url, api_key, model, messages, temperature)

    return answer, chunks


def query_without_rag(
    question: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: Optional[float] = None,
) -> Optional[str]:
    """Query without RAG: send question directly to LLM."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
    ]

    console.print("[dim]Sending query to LLM without RAG context...[/dim]")
    answer = _call_llm(base_url, api_key, model, messages, temperature)

    return answer


def compare_answers(
    question: str,
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    model: str,
    k: int = 5,
    temperature: Optional[float] = None,
) -> None:
    """Compare answers with and without RAG."""
    console.print()
    console.print(f"[bold cyan]Question:[/bold cyan] {question}")
    console.print()

    rag_answer, chunks = query_with_rag(question, index, api_key, base_url, model, k, temperature)
    console.print()
    non_rag_answer = query_without_rag(question, api_key, base_url, model, temperature)
    console.print()

    console.print("[bold]Comparison Results:[/bold]")
    console.print()

    if chunks:
        console.print("[bold]Retrieved Chunks:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Document", style="white")
        table.add_column("Chunk", style="dim", width=8)
        table.add_column("Distance", style="yellow", width=10)
        table.add_column("Preview", style="dim", width=50)

        for chunk in chunks:
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

    panels = []
    if rag_answer:
        rag_panel = Panel(
            Markdown(rag_answer, code_theme="monokai"),
            title="[bold green]With RAG[/bold green]",
            border_style="green",
        )
        panels.append(rag_panel)
    else:
        panels.append(Panel("[red]Failed to get RAG answer[/red]", title="[bold red]With RAG[/bold red]", border_style="red"))

    if non_rag_answer:
        non_rag_panel = Panel(
            Markdown(non_rag_answer, code_theme="monokai"),
            title="[bold yellow]Without RAG[/bold yellow]",
            border_style="yellow",
        )
        panels.append(non_rag_panel)
    else:
        panels.append(Panel("[red]Failed to get non-RAG answer[/red]", title="[bold red]Without RAG[/bold red]", border_style="red"))

    console.print(Columns(panels, equal=True, expand=True))
    console.print()


def print_conclusion() -> None:
    """Print conclusion about RAG effectiveness."""
    conclusion = """
## RAG Analysis Conclusion

### Where RAG Helps:

1. **Factual Accuracy**: RAG provides access to specific, up-to-date information from indexed documents, reducing hallucinations and improving accuracy for domain-specific questions.

2. **Context-Specific Answers**: When questions relate to content in the indexed documents, RAG ensures answers are grounded in the actual source material rather than general knowledge.

3. **Detailed Information**: RAG can provide more detailed and specific answers by retrieving relevant context that the model might not have in its training data.

4. **Source Attribution**: RAG allows you to see which documents and chunks were used, providing transparency and traceability.

### Where RAG May Not Help:

1. **General Knowledge Questions**: For questions about well-known facts, general knowledge, or topics not in the indexed documents, RAG may not provide additional value.

2. **Creative or Abstract Questions**: Questions requiring creative thinking, reasoning, or abstract concepts may not benefit from document retrieval.

3. **Poor Index Quality**: If the document index is incomplete, outdated, or doesn't contain relevant information, RAG won't improve answers.

4. **Irrelevant Retrievals**: If the search retrieves chunks that aren't actually relevant to the question, it can add noise and potentially confuse the model.

### Key Takeaways:

- RAG is most effective when you have a well-indexed corpus of relevant documents
- The quality of retrieved chunks directly impacts answer quality
- RAG shines for domain-specific, factual, or document-based questions
- For general knowledge, the base model may perform similarly without RAG
"""
    console.print(Panel(Markdown(conclusion), title="[bold cyan]Conclusion[/bold cyan]", border_style="cyan"))


def run() -> None:
    """Main entry point for Day 14 task."""
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 14 — First RAG Query[/bold cyan]")
    console.print()
    console.print("[dim]This tool compares LLM answers with and without RAG.[/dim]")
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
        console.print("  1. Ask a question (compare with/without RAG)")
        console.print("  2. View conclusion about RAG effectiveness")
        console.print("  3. Exit")
        console.print()

        choice = Prompt.ask("Select an option", default="3")

        if choice == "1":
            question = Prompt.ask("Enter your question")
            if not question:
                continue

            k = int(Prompt.ask("Number of chunks to retrieve", default="5"))

            compare_answers(question, index, api_key, base_url, model, k=k)

        elif choice == "2":
            print_conclusion()

        elif choice == "3":
            console.print("[yellow]Goodbye![/yellow]")
            break

        else:
            console.print("[yellow]Invalid option. Please try again.[/yellow]")


