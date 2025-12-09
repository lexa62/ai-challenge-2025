import os
from pathlib import Path
from typing import Optional

import requests


class TranscriptionError(Exception):
    pass


def _build_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
    }


def _get_audio_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/audio/transcriptions"


def transcribe_file(
    path: str,
    model: str,
    api_key: str,
    base_url: str,
    language: Optional[str] = None,
) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        raise TranscriptionError(f"File not found: {file_path}")

    if not api_key:
        raise TranscriptionError("OPENAI_API_KEY is required for transcription.")

    url = _get_audio_endpoint(base_url)
    headers = _build_headers(api_key)

    data = {
        "model": model,
        "response_format": "text",
    }
    if language:
        data["language"] = language

    try:
        with file_path.open("rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = requests.post(url, headers=headers, data=data, files=files, timeout=600)
    except requests.Timeout as e:
        raise TranscriptionError("Transcription request timed out.") from e
    except requests.RequestException as e:
        raise TranscriptionError(f"Network error during transcription: {e}") from e

    if response.status_code != 200:
        snippet = ""
        try:
            err_json = response.json()
            snippet = err_json.get("error", {}).get("message", "") or response.text[:300]
        except Exception:
            snippet = response.text[:300]
        raise TranscriptionError(f"Transcription failed (HTTP {response.status_code}): {snippet}")

    text = response.text.strip()
    if not text:
        raise TranscriptionError("Transcription succeeded but returned empty text.")

    return text

