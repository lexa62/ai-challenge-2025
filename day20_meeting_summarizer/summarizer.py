import math
import re
from typing import Any, Dict, List

import requests


class SummarizationError(Exception):
    pass


def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _get_chat_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def _post_chat_completions(
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    url = _get_chat_endpoint(base_url)
    headers = _build_headers(api_key)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _extract_message_content(body: Dict[str, Any]) -> str:
    try:
        return (body["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""


def _chunk_transcript(text: str, max_chars: int = 8000) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        if not sentence:
            continue
        if current_len + len(sentence) + 1 > max_chars and current:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def _build_chunk_prompt(brief: bool) -> str:
    detail_instruction = (
        "Be concise but capture all important decisions and action items."
        if brief
        else "Provide a detailed but focused summary."
    )
    return (
        "You are an assistant summarizing a segment of a meeting transcript.\n"
        f"{detail_instruction}\n\n"
        "For this segment only, return a short markdown summary with:\n"
        "- Key points\n"
        "- Decisions (if mentioned)\n"
        "- Action items with owners and due dates when available\n"
        "- Open questions or risks\n"
    )


def _build_final_system_prompt(brief: bool) -> str:
    detail_instruction = (
        "Make each section as short and to-the-point as possible while keeping important decisions and actions."
        if brief
        else "Provide a clear, structured, and moderately detailed business-style summary."
    )
    return (
        "You are an expert meeting notes assistant.\n"
        f"{detail_instruction}\n\n"
        "You will receive one or more partial summaries of a meeting.\n"
        "Synthesize them into a single, coherent markdown document with the following sections:\n"
        "1. Title\n"
        "2. Overview (3–5 sentences)\n"
        "3. Key Points (bullet list)\n"
        "4. Decisions (bullet list)\n"
        "5. Action Items (bullet list with owner and due date when possible)\n"
        "6. Open Questions / Risks (bullet list)\n\n"
        "If some sections are not applicable, still include the heading and write 'None noted'."
    )


def _call_chat_model(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1500,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    try:
        body = _post_chat_completions(base_url, api_key, payload, timeout_seconds=120.0)
    except requests.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else "unknown"
        snippet = ""
        try:
            err_json = http_err.response.json() if http_err.response is not None else {}
            snippet = err_json.get("error", {}).get("message", "") or http_err.response.text[:300]
        except Exception:
            snippet = ""
        raise SummarizationError(f"Chat completion failed (HTTP {status}): {snippet}") from http_err
    except requests.RequestException as req_err:
        raise SummarizationError(f"Network error during chat completion: {req_err}") from req_err

    content = _extract_message_content(body)
    if not content:
        raise SummarizationError("Chat completion returned no content.")
    return content


def summarize_transcript(
    transcript: str,
    model: str,
    api_key: str,
    base_url: str,
    brief: bool = False,
) -> str:
    transcript = (transcript or "").strip()
    if not transcript:
        raise SummarizationError("Transcript is empty.")

    chunks = _chunk_transcript(transcript)

    if not chunks:
        raise SummarizationError("Transcript could not be chunked.")

    if len(chunks) == 1:
        system_prompt = _build_final_system_prompt(brief)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Here is a full meeting transcript. Produce the structured meeting notes as instructed.\n\n"
                + chunks[0],
            },
        ]
        return _call_chat_model(base_url, api_key, model, messages)

    partial_summaries: List[str] = []
    chunk_system_prompt = _build_chunk_prompt(brief)

    for idx, chunk in enumerate(chunks, start=1):
        messages = [
            {"role": "system", "content": chunk_system_prompt},
            {
                "role": "user",
                "content": f"Meeting transcript segment {idx} of {len(chunks)}:\n\n{chunk}",
            },
        ]
        summary = _call_chat_model(base_url, api_key, model, messages, max_tokens=800)
        partial_summaries.append(summary)

    combined = "\n\n---\n\n".join(
        f"Segment {i+1} summary:\n{partial_summaries[i]}" for i in range(len(partial_summaries))
    )

    final_system_prompt = _build_final_system_prompt(brief)
    final_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": final_system_prompt},
        {
            "role": "user",
            "content": "Here are summaries of different segments of the same meeting. "
            "Combine them into a single structured meeting summary as instructed.\n\n"
            + combined,
        },
    ]

    return _call_chat_model(base_url, api_key, model, final_messages)

