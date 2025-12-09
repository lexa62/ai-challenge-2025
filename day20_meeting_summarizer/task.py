import argparse
import os
from pathlib import Path
from typing import Optional

from rich.panel import Panel

from utils.config import settings
from utils.helpers import console, header, info, success, warn

from day20_meeting_summarizer.transcription import transcribe_file
from day20_meeting_summarizer.summarizer import summarize_transcript


def _get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="day20_meeting_summarizer",
        description="Transcribe an audio/video file and generate a structured meeting summary.",
    )
    parser.add_argument(
        "file_path",
        help="Path to the audio or video file (e.g. .mp3, .wav, .m4a, .mp4)",
    )
    parser.add_argument(
        "--language",
        "-l",
        help="Language hint for transcription (e.g. 'en', 'de'). Defaults to automatic detection.",
        default=None,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to save the summary as a Markdown file.",
        default=None,
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Override chat model for summarization (defaults to OPENAI_MODEL).",
        default=None,
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Generate a shorter, more condensed summary.",
    )
    return parser.parse_args(argv)


def _write_output(summary: str, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")
    info(f"Summary saved to {path}")


def run(argv: Optional[list[str]] = None) -> None:
    """Main entry point for Day 20 Meeting Notes Summarizer."""
    header("Day 20 — Meeting Notes Summarizer")

    api_key = settings.openai_api_key
    if not api_key:
        warn("OPENAI_API_KEY is not set. Add it to your .env and try again.")
        return

    args = _parse_args(argv)

    file_path = Path(args.file_path)
    if not file_path.is_file():
        warn(f"Input file not found: {file_path}")
        return

    audio_model = getattr(settings, "openai_audio_model", None) or os.getenv(
        "OPENAI_AUDIO_MODEL", "gpt-4o-mini-transcribe"
    )
    chat_model = args.model or settings.openai_model
    base_url = _get_base_url()

    info(f"Using audio model: {audio_model}")
    info(f"Using chat model: {chat_model}")
    info(f"Base URL: {base_url}")
    console.print()

    try:
        info("Transcribing audio/video file...")
        transcript = transcribe_file(
            path=str(file_path),
            model=audio_model,
            api_key=api_key,
            base_url=base_url,
            language=args.language,
        )
    except Exception as e:
        warn(f"Transcription failed: {e}")
        return

    if not transcript or not transcript.strip():
        warn("Transcription produced no text. Nothing to summarize.")
        return

    console.print()
    info("Generating meeting summary with LLM...")

    try:
        summary = summarize_transcript(
            transcript=transcript,
            model=chat_model,
            api_key=api_key,
            base_url=base_url,
            brief=args.brief,
        )
    except Exception as e:
        warn(f"Summarization failed: {e}")
        return

    if not summary or not summary.strip():
        warn("Summarization produced no output.")
        return

    console.print()
    console.print(
        Panel.fit(
            summary,
            title="Meeting Summary",
            border_style="green",
        )
    )

    if args.output:
        _write_output(summary, args.output)

    console.print()
    success("Summary generated!")
    console.print()


if __name__ == "__main__":
    run()
