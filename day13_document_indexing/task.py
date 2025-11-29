import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import faiss
except ImportError:
    faiss = None

from utils.config import settings
from utils.helpers import console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com")


def _get_embedding(text: str, api_key: str, base_url: str) -> Optional[List[float]]:
    """Generate embedding for a text using OpenAI API."""
    url = f"{base_url.rstrip('/')}/v1/embeddings"
    headers = _build_headers(api_key)

    payload = {
        "model": "text-embedding-3-small",
        "input": text,
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        console.print(f"[red]Error generating embedding: {e}[/red]")
        return None


def load_document(file_path: Path) -> Optional[str]:
    """Load text content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            console.print(f"[red]Error reading {file_path}: {e}[/red]")
            return None
    except Exception as e:
        console.print(f"[red]Error reading {file_path}: {e}[/red]")
        return None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        chunk_end = end
        if end < len(text):
            last_period = text.rfind(".", start, end)
            last_newline = text.rfind("\n", start, end)

            if last_period > start and last_period > last_newline:
                chunk_end = last_period + 1
            elif last_newline > start:
                chunk_end = last_newline + 1

        chunks.append(text[start:chunk_end])
        start = chunk_end - overlap
        if start < 0:
            start = 0

    return chunks


class DocumentIndex:
    """Manages document indexing with embeddings and FAISS."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.metadata_path = index_path.parent / f"{index_path.stem}_metadata.json"
        self.index: Optional[Any] = None
        self.metadata: List[Dict[str, Any]] = []
        self.dimension: int = 1536

    def _initialize_faiss(self) -> bool:
        """Initialize FAISS index."""
        if faiss is None:
            console.print("[red]FAISS is not installed. Install it with: pip install faiss-cpu[/red]")
            return False

        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.dimension = self.index.d
                console.print(f"[green]Loaded existing index with {self.index.ntotal} vectors[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load existing index: {e}. Creating new one.[/yellow]")
                self.index = faiss.IndexFlatL2(self.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            console.print(f"[green]Created new FAISS index with dimension {self.dimension}[/green]")

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                console.print(f"[green]Loaded {len(self.metadata)} metadata entries[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not load metadata: {e}[/yellow]")
                self.metadata = []

        return True

    def add_documents(
        self,
        documents: List[Tuple[str, str]],
        api_key: str,
        base_url: str,
    ) -> int:
        """Add documents to the index."""
        if self.index is None:
            if not self._initialize_faiss():
                return 0

        added_count = 0
        embeddings = []
        new_metadata = []

        console.print("[dim]Generating embeddings...[/dim]")

        for doc_id, text in documents:
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                embedding = _get_embedding(chunk, api_key, base_url)
                if embedding is None:
                    continue

                embeddings.append(embedding)
                new_metadata.append({
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "chunk_text": chunk,
                    "total_chunks": len(chunks),
                })
                added_count += 1

        if embeddings:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            self.index.add(embeddings_array)
            self.metadata.extend(new_metadata)
            console.print(f"[green]Added {added_count} chunks to index[/green]")

        return added_count

    def search(self, query: str, api_key: str, base_url: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index is None:
            if not self._initialize_faiss():
                return []

        if self.index.ntotal == 0:
            console.print("[yellow]Index is empty. Please index some documents first.[/yellow]")
            return []

        query_embedding = _get_embedding(query, api_key, base_url)
        if query_embedding is None:
            return []

        query_vector = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result["distance"] = float(distance)
                result["rank"] = i + 1
                results.append(result)

        return results

    def save(self) -> bool:
        """Save index and metadata to disk."""
        if self.index is None:
            return False

        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            console.print(f"[green]Saved index to {self.index_path}[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Error saving index: {e}[/red]")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if self.index is None:
            return {"total_vectors": 0, "dimension": 0, "documents": 0}

        doc_ids = set(m["doc_id"] for m in self.metadata)
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "documents": len(doc_ids),
        }


def _collect_documents(directory: Path, extensions: List[str]) -> List[Path]:
    """Collect all documents from a directory."""
    documents = []

    for ext in extensions:
        pattern = f"**/*{ext}"
        documents.extend(directory.glob(pattern))

    return sorted(documents)


def _index_directory(
    directory: Path,
    index: DocumentIndex,
    api_key: str,
    base_url: str,
    extensions: List[str],
) -> int:
    """Index all documents in a directory."""
    console.print(f"[cyan]Scanning directory: {directory}[/cyan]")
    files = _collect_documents(directory, extensions)

    if not files:
        console.print(f"[yellow]No documents found in {directory}[/yellow]")
        return 0

    console.print(f"[green]Found {len(files)} files[/green]")

    documents = []
    for file_path in files:
        console.print(f"[dim]Loading: {file_path.name}[/dim]")
        content = load_document(file_path)
        if content:
            documents.append((str(file_path), content))

    if documents:
        return index.add_documents(documents, api_key, base_url)
    return 0


def run() -> None:
    """Main entry point for Day 13 task."""
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()

    console.print("[bold cyan]Day 13 — Document Indexing[/bold cyan]")
    console.print()
    console.print("[dim]This tool creates a local document index with embeddings for retrieval.[/dim]")
    console.print()

    index_dir = Path("outputs")
    index_dir.mkdir(exist_ok=True)
    index_path = index_dir / "document_index.faiss"
    index = DocumentIndex(index_path)

    if not index._initialize_faiss():
        console.print("[red]Failed to initialize FAISS. Please install faiss-cpu: pip install faiss-cpu[/red]")
        return

    while True:
        console.print()
        console.print("[bold]Options:[/bold]")
        console.print("  1. Index documents from directory")
        console.print("  2. Search indexed documents")
        console.print("  3. View index statistics")
        console.print("  4. Save index")
        console.print("  5. Exit")
        console.print()

        choice = Prompt.ask("Select an option", default="5")

        if choice == "1":
            dir_path = Prompt.ask("Enter directory path", default=".")
            directory = Path(dir_path)

            if not directory.exists():
                console.print(f"[red]Directory not found: {directory}[/red]")
                continue

            extensions_input = Prompt.ask(
                "Enter file extensions (comma-separated)",
                default=".txt,.md,.py,.js,.ts,.json,.yaml,.yml"
            )
            extensions = [ext.strip() if ext.strip().startswith(".") else f".{ext.strip()}"
                         for ext in extensions_input.split(",")]

            console.print()
            added = _index_directory(directory, index, api_key, base_url, extensions)
            console.print(f"[green]Indexed {added} chunks from {directory}[/green]")

            if Confirm.ask("Save index now?"):
                index.save()

        elif choice == "2":
            query = Prompt.ask("Enter search query")
            if not query:
                continue

            k = int(Prompt.ask("Number of results", default="5"))

            console.print()
            console.print(f"[cyan]Searching for: {query}[/cyan]")
            console.print()

            results = index.search(query, api_key, base_url, k=k)

            if not results:
                console.print("[yellow]No results found.[/yellow]")
                continue

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Rank", style="cyan", width=6)
            table.add_column("Document", style="white")
            table.add_column("Chunk", style="dim", width=8)
            table.add_column("Distance", style="yellow", width=10)
            table.add_column("Preview", style="dim", width=50)

            for result in results:
                doc_path = Path(result["doc_id"])
                preview = result["chunk_text"][:100].replace("\n", " ")
                if len(result["chunk_text"]) > 100:
                    preview += "..."

                table.add_row(
                    str(result["rank"]),
                    doc_path.name,
                    f"{result['chunk_index'] + 1}/{result['total_chunks']}",
                    f"{result['distance']:.4f}",
                    preview,
                )

            console.print(table)
            console.print()

            if Confirm.ask("Show full text of a result?"):
                rank = int(Prompt.ask("Enter rank number", default="1"))
                if 1 <= rank <= len(results):
                    result = results[rank - 1]
                    console.print()
                    console.print(f"[bold]Document:[/bold] {result['doc_id']}")
                    console.print(f"[bold]Chunk {result['chunk_index'] + 1} of {result['total_chunks']}:[/bold]")
                    console.print()
                    md = Markdown(result["chunk_text"])
                    console.print(Panel(md, border_style="green"))

        elif choice == "3":
            stats = index.get_stats()
            console.print()
            console.print("[bold]Index Statistics:[/bold]")
            console.print(f"  Total vectors: {stats['total_vectors']}")
            console.print(f"  Dimension: {stats['dimension']}")
            console.print(f"  Unique documents: {stats['documents']}")

        elif choice == "4":
            if index.save():
                console.print("[green]Index saved successfully![/green]")
            else:
                console.print("[red]Failed to save index.[/red]")

        elif choice == "5":
            if index.index and index.index.ntotal > 0:
                if Confirm.ask("Save index before exiting?"):
                    index.save()
            console.print("[yellow]Goodbye![/yellow]")
            break

        else:
            console.print("[yellow]Invalid option. Please try again.[/yellow]")

