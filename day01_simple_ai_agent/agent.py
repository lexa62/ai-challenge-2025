import json
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.config import settings
from utils.helpers import console
from rich.markdown import Markdown
from rich.panel import Panel


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


def run_cli(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = "You are a helpful assistant.",
) -> None:
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    active_model = model or settings.openai_model

    console.print(f"[dim]Model:[/dim] {active_model}  [dim]Base URL:[/dim] {base_url}")
    console.print("[dim]Type ':q' to quit[/dim]")

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Bye.[/yellow]")
            break

        if user_input in {"", ":q", ":quit"}:
            console.print("[yellow]Goodbye![/yellow]")
            break

        messages.append({"role": "user", "content": user_input})

        payload = _build_payload(
            model=active_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            use_completion_tokens=False,
        )

        try:
            body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=30.0)
        except requests.Timeout:
            console.print("[red]Request timed out. Try again or adjust your prompt.[/red]")
            continue
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
            # Retry with max_completion_tokens for models that require it
            if status == 400 and ("max_tokens" in (snippet or "") or "max_tokens" in (http_err.response.text if http_err.response is not None else "")):
                try:
                    payload_retry = _build_payload(
                        model=active_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        use_completion_tokens=True,
                    )
                    body = _post_chat_completions(base_url, api_key, payload_retry, timeout_seconds=30.0)
                except Exception:
                    console.print(f"[red]HTTP {status}[/red] {snippet}")
                    continue
            else:
                console.print(f"[red]HTTP {status}[/red] {snippet}")
                continue
        except requests.RequestException as req_err:
            console.print(f"[red]Network error:[/red] {req_err}")
            continue
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON response from server.[/red]")
            continue

        assistant = _extract_message_content(body)
        if not assistant:
            console.print("[yellow]No content returned.[/yellow]")
            continue

        messages.append({"role": "assistant", "content": assistant})
        md = Markdown(assistant, code_theme="monokai", justify="left")
        console.print(Panel.fit(md, border_style="green"))

