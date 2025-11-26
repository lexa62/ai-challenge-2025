import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from utils.config import settings
from utils.helpers import console
from rich.markdown import Markdown
from rich.panel import Panel

from .memory import MemoryManager


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
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    use_completion_tokens: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        key = "max_completion_tokens" if use_completion_tokens else "max_tokens"
        payload[key] = max_tokens
    return payload


def _send_message_and_get_response(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    user_message: str,
    memory_manager: MemoryManager,
    session_id: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> bool:
    user_msg = {"role": "user", "content": user_message}
    messages.append(user_msg)

    payload = _build_payload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        use_completion_tokens=False,
    )

    try:
        body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=30.0)
    except requests.Timeout:
        console.print("[red]Request timed out. Try again or adjust your prompt.[/red]")
        memory_manager.save_messages(session_id, [user_msg])
        return False
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
        if status == 400 and ("max_tokens" in (snippet or "") or "max_tokens" in (http_err.response.text if http_err.response is not None else "")):
            try:
                payload_retry = _build_payload(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    use_completion_tokens=True,
                )
                body = _post_chat_completions(base_url, api_key, payload_retry, timeout_seconds=30.0)
            except Exception:
                console.print(f"[red]HTTP {status}[/red] {snippet}")
                memory_manager.save_messages(session_id, [user_msg])
                return False
        else:
            console.print(f"[red]HTTP {status}[/red] {snippet}")
            memory_manager.save_messages(session_id, [user_msg])
            return False
    except requests.RequestException as req_err:
        console.print(f"[red]Network error:[/red] {req_err}")
        memory_manager.save_messages(session_id, [user_msg])
        return False
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON response from server.[/red]")
        memory_manager.save_messages(session_id, [user_msg])
        return False

    assistant_message = body["choices"][0]["message"]
    assistant_content = _extract_message_content(body)

    if not assistant_content:
        console.print("[yellow]No content returned.[/yellow]")
        memory_manager.save_messages(session_id, [user_msg])
        return False

    messages.append(assistant_message)
    md = Markdown(assistant_content, code_theme="monokai", justify="left")
    console.print(Panel.fit(md, border_style="green"))

    memory_manager.save_messages(session_id, [user_msg, assistant_message])
    return True


def run() -> None:
    """Main entry point for Day 11 task with external memory."""
    import sys

    if "--clear-memory" in sys.argv or "-c" in sys.argv:
        db_path = Path(__file__).parent / "memory.db"
        if db_path.exists():
            with MemoryManager(str(db_path)) as memory_manager:
                memory_manager.clear_messages()
            console.print("[green]✓ Memory database cleared[/green]")
        else:
            console.print("[yellow]No memory database found to clear[/yellow]")
        return

    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 11 — External Memory for Conversations[/bold cyan]")
    console.print()

    db_path = Path(__file__).parent / "memory.db"
    session_id = "default"

    messages: List[Dict[str, Any]] = []

    with MemoryManager(str(db_path)) as memory_manager:
        loaded_messages = memory_manager.load_messages(session_id)
        message_count = memory_manager.get_message_count(session_id)

        if loaded_messages:
            messages = loaded_messages
            has_system = any(msg.get("role") == "system" for msg in messages)
            if not has_system:
                system_prompt = "You are a helpful assistant with memory of previous conversations."
                messages.insert(0, {"role": "system", "content": system_prompt})
            console.print(f"[green]✓ Loaded {len(loaded_messages)} messages from previous session[/green]")
            console.print(f"[dim]Total messages in database: {message_count}[/dim]")
        else:
            console.print("[dim]No previous conversation found. Starting fresh.[/dim]")
            system_prompt = "You are a helpful assistant with memory of previous conversations."
            messages.append({"role": "system", "content": system_prompt})
            memory_manager.save_messages(session_id, messages)

        console.print()
        console.print(f"[dim]Model:[/dim] {model}  [dim]Base URL:[/dim] {base_url}")
        console.print(f"[dim]Memory DB:[/dim] {db_path}")
        console.print("[dim]Type ':q' to quit[/dim]")
        console.print()

        if len(loaded_messages) > 0:
            console.print("[dim]Continuing previous conversation...[/dim]")
            console.print()

        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[yellow]Bye.[/yellow]")
                break

            if user_input in {"", ":q", ":quit"}:
                console.print("[yellow]Goodbye![/yellow]")
                break

            _send_message_and_get_response(
                base_url,
                api_key,
                model,
                messages,
                user_input,
                memory_manager,
                session_id,
            )
            console.print()

