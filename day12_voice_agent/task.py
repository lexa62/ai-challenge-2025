import json
import math
import os
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    import speech_recognition as sr
    import pyaudio
except ImportError:
    sr = None
    pyaudio = None

from utils.config import settings
from utils.helpers import console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.table import Table
from rich.console import Group


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


def _send_message_to_llm(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    user_message: str,
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
                return False
        else:
            console.print(f"[red]HTTP {status}[/red] {snippet}")
            return False
    except requests.RequestException as req_err:
        console.print(f"[red]Network error:[/red] {req_err}")
        return False
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON response from server.[/red]")
        return False

    assistant_message = body["choices"][0]["message"]
    assistant_content = _extract_message_content(body)

    if not assistant_content:
        console.print("[yellow]No content returned.[/yellow]")
        return False

    messages.append(assistant_message)
    md = Markdown(assistant_content, code_theme="monokai", justify="left")
    console.print(Panel.fit(md, border_style="green"))

    return True


def _calculate_rms(data: bytes, sample_width: int) -> float:
    """Calculate RMS (Root Mean Square) of audio data."""
    if sample_width == 1:
        fmt = "%dB" % len(data)
        samples = struct.unpack(fmt, data)
    elif sample_width == 2:
        fmt = "%dh" % (len(data) // 2)
        samples = struct.unpack(fmt, data)
    else:
        return 0.0

    if not samples:
        return 0.0

    sum_squares = sum(s * s for s in samples)
    rms = math.sqrt(sum_squares / len(samples))
    return rms


def _show_audio_levels(microphone: Any, stop_event: threading.Event) -> None:
    """Display real-time audio levels in the terminal."""
    if pyaudio is None:
        return

    stream = None
    p = None

    try:
        # Use standard sample rate for speech recognition (16kHz)
        sample_rate = 16000
        p = pyaudio.PyAudio()

        # Get device index from microphone
        device_index = None
        if hasattr(microphone, 'device_index') and microphone.device_index is not None:
            device_index = microphone.device_index

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
                stream_callback=None,
            )
        except OSError:
            # Device might be in use, try without specifying device
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024,
                    stream_callback=None,
                )
            except Exception:
                return

        max_level = 0
        levels_history = []
        history_size = 50

        with Live(console=console, refresh_per_second=10) as live:
            while not stop_event.is_set():
                try:
                    data = stream.read(1024, exception_on_overflow=False)
                    rms = _calculate_rms(data, 2)

                    # Normalize to 0-100 scale
                    # Adjust sensitivity: lower divisor = more sensitive
                    # For 16-bit audio, max RMS is around 32768/sqrt(2) ≈ 23170
                    # Scale to make normal speech show 20-80%
                    normalized_level = min(100, (rms / 500) * 100)
                    max_level = max(max_level, normalized_level)

                    levels_history.append(normalized_level)
                    if len(levels_history) > history_size:
                        levels_history.pop(0)

                    # Create visual bar
                    bar_width = 50
                    current_bar = int((normalized_level / 100) * bar_width)

                    # Build visualization using Rich Text objects for proper rendering
                    status_icon = "🔊" if normalized_level > 10 else "🔇"
                    header = Text(f"{status_icon} Listening...", style="bold cyan")

                    # Main waveform visualization
                    waveform_width = 50
                    waveform_text = Text("  ")
                    if len(levels_history) >= 10:
                        step = max(1, len(levels_history) // waveform_width)
                        for i in range(0, len(levels_history), step):
                            level = levels_history[i]
                            height = int((level / 100) * 8)

                            # Choose character and color
                            if height >= 7:
                                char = "█"
                                style = "bright_red"
                            elif height >= 6:
                                char = "▇"
                                style = "red"
                            elif height >= 5:
                                char = "▆"
                                style = "yellow"
                            elif height >= 4:
                                char = "▅"
                                style = "bright_yellow"
                            elif height >= 3:
                                char = "▄"
                                style = "green"
                            elif height >= 2:
                                char = "▃"
                                style = "bright_green"
                            elif height >= 1:
                                char = "▂"
                                style = "dim white"
                            else:
                                char = "▁"
                                style = "dim"

                            waveform_text.append(char, style=style)

                    # Current level bar
                    bar_text = Text("  ")
                    for i in range(bar_width):
                        if i < current_bar:
                            pos_ratio = i / bar_width if bar_width > 0 else 0
                            if pos_ratio < 0.3:
                                bar_text.append("█", style="green")
                            elif pos_ratio < 0.6:
                                bar_text.append("█", style="yellow")
                            else:
                                bar_text.append("█", style="red")
                        else:
                            bar_text.append("░", style="dim")

                    # Percentage text
                    if normalized_level > 70:
                        pct_style = "bright_red"
                    elif normalized_level > 50:
                        pct_style = "yellow"
                    elif normalized_level > 25:
                        pct_style = "green"
                    elif normalized_level > 10:
                        pct_style = "bright_green"
                    else:
                        pct_style = "dim white"

                    bar_text.append(f" {normalized_level:5.1f}%", style=pct_style)

                    # Peak indicator
                    peak_text = Text()
                    if max_level > 0 and max_level > normalized_level:
                        peak_text = Text(f"  📈 Peak: {max_level:.1f}%", style="dim")

                    # Combine all elements
                    display_group = Group(
                        header,
                        Text(),
                        waveform_text,
                        Text(),
                        bar_text,
                        peak_text if peak_text else Text()
                    )

                    live.update(display_group)
                    time.sleep(0.1)

                except Exception:
                    break

        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if p:
            try:
                p.terminate()
            except Exception:
                pass
    except Exception:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
        if p:
            try:
                p.terminate()
            except Exception:
                pass


def _listen_for_speech(recognizer: Any, microphone: Any, timeout: float = 5.0, phrase_time_limit: float = 10.0) -> Optional[str]:
    """Listen for speech and convert to text using Google's free API with visual feedback."""
    stop_event = threading.Event()
    audio_levels_thread = None

    try:
        with microphone as source:
            console.print("[dim]Adjusting for ambient noise...[/dim]")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            # Start audio level visualization in a separate thread
            if pyaudio is not None:
                audio_levels_thread = threading.Thread(
                    target=_show_audio_levels,
                    args=(microphone, stop_event),
                    daemon=True
                )
                audio_levels_thread.start()
                time.sleep(0.2)  # Give visualization time to start

            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            stop_event.set()

    except sr.WaitTimeoutError:
        stop_event.set()
        console.print("\n[yellow]No speech detected. Please try again.[/yellow]")
        return None
    except sr.RequestError as e:
        stop_event.set()
        console.print(f"\n[red]Could not request results from speech recognition service: {e}[/red]")
        return None
    except Exception as e:
        stop_event.set()
        console.print(f"\n[red]Error capturing audio: {e}[/red]")
        return None
    finally:
        stop_event.set()
        if audio_levels_thread and audio_levels_thread.is_alive():
            time.sleep(0.2)  # Give thread time to clean up

    try:
        console.print("\n[dim]Processing speech...[/dim]")
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        console.print("[yellow]Could not understand audio. Please try again.[/yellow]")
        return None
    except sr.RequestError as e:
        console.print(f"[red]Could not request results from Google Speech Recognition service: {e}[/red]")
        return None
    except Exception as e:
        console.print(f"[red]Error processing speech: {e}[/red]")
        return None


def _list_audio_devices() -> List[Dict[str, Any]]:
    """List all available audio input devices."""
    if pyaudio is None:
        return []

    devices = []
    try:
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()

        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': info.get('name', 'Unknown'),
                        'channels': info.get('maxInputChannels', 0),
                        'sample_rate': int(info.get('defaultSampleRate', 44100)),
                    })
            except Exception:
                continue

        p.terminate()
    except Exception:
        pass

    return devices


def _select_microphone() -> Optional[int]:
    """Display available microphones and let user select one."""
    devices = _list_audio_devices()

    if not devices:
        console.print("[yellow]No audio input devices found.[/yellow]")
        return None

    console.print()
    console.print("[bold cyan]Available Microphones:[/bold cyan]")
    console.print()

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Name", style="white")
    table.add_column("Channels", style="dim", width=10)
    table.add_column("Sample Rate", style="dim", width=12)

    for i, device in enumerate(devices, 1):
        table.add_row(
            str(i),
            device['name'],
            str(device['channels']),
            f"{device['sample_rate']} Hz"
        )

    console.print(table)
    console.print()
    console.print("[dim]Enter microphone number (or press Enter for default):[/dim]")

    try:
        choice = input("> ").strip()
        if not choice:
            return None

        device_num = int(choice)
        if 1 <= device_num <= len(devices):
            selected = devices[device_num - 1]
            console.print(f"[green]Selected: {selected['name']}[/green]")
            console.print()
            return selected['index']
        else:
            console.print("[yellow]Invalid selection. Using default microphone.[/yellow]")
            console.print()
            return None
    except (ValueError, KeyboardInterrupt, EOFError):
        console.print("[yellow]Using default microphone.[/yellow]")
        console.print()
        return None


def _initialize_speech_recognition(device_index: Optional[int] = None) -> Tuple[Optional[Any], Optional[Any]]:
    """Initialize speech recognition and microphone."""
    if sr is None:
        console.print("[red]speech_recognition library is not installed.[/red]")
        console.print("[yellow]Install it with: pip install SpeechRecognition pyaudio[/yellow]")
        console.print("[yellow]Note: On macOS, you may need to install PortAudio first: brew install portaudio[/yellow]")
        return None, None

    try:
        recognizer = sr.Recognizer()
        if device_index is not None:
            microphone = sr.Microphone(device_index=device_index)
        else:
            microphone = sr.Microphone()
        return recognizer, microphone
    except OSError as e:
        console.print(f"[red]Could not access microphone: {e}[/red]")
        console.print("[yellow]Make sure your microphone is connected and permissions are granted.[/yellow]")
        return None, None
    except Exception as e:
        console.print(f"[red]Error initializing speech recognition: {e}[/red]")
        return None, None


def run() -> None:
    """Main entry point for Day 12 task with voice agent."""
    api_key = settings.openai_api_key
    if not api_key:
        console.print("[red]OPENAI_API_KEY is not set. Add it to your .env and try again.[/red]")
        return

    base_url = _get_base_url()
    model = settings.openai_model

    console.print("[bold cyan]Day 12 — Voice Agent (Speech → LLM → Text)[/bold cyan]")
    console.print()

    # Allow user to select microphone
    device_index = _select_microphone()

    recognizer, microphone = _initialize_speech_recognition(device_index=device_index)
    if recognizer is None or microphone is None:
        console.print("[yellow]Falling back to text input mode...[/yellow]")
        console.print()
        use_text_input = True
    else:
        use_text_input = False

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful assistant that responds to voice commands. Keep responses concise and clear."
        }
    ]

    console.print(f"[dim]Model:[/dim] {model}  [dim]Base URL:[/dim] {base_url}")
    if not use_text_input:
        console.print("[dim]Voice mode: Speak your query (or type ':q' to quit, ':t' to toggle text mode)[/dim]")
    else:
        console.print("[dim]Text mode: Type your query (or ':q' to quit)[/dim]")
    console.print()

    while True:
        try:
            if use_text_input:
                user_input = input("> ").strip()
                if user_input in {"", ":q", ":quit"}:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                if user_input == ":t":
                    if recognizer is not None and microphone is not None:
                        use_text_input = False
                        console.print("[green]Switched to voice mode[/green]")
                        console.print()
                    else:
                        console.print("[yellow]Voice mode not available. Microphone access failed.[/yellow]")
                        console.print()
                    continue
            else:
                user_input = _listen_for_speech(recognizer, microphone, timeout=5.0, phrase_time_limit=10.0)
                if user_input is None:
                    console.print()
                    continue

                if user_input.lower() in {"quit", "exit", "goodbye", "bye"}:
                    console.print("[yellow]Goodbye![/yellow]")
                    break

                if user_input.lower() in {"text mode", "switch to text", "toggle text"}:
                    use_text_input = True
                    console.print("[green]Switched to text mode[/green]")
                    console.print()
                    continue

                console.print(f"[bold]You said:[/bold] {user_input}")
                console.print()

        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Bye.[/yellow]")
            break

        if user_input:
            _send_message_to_llm(
                base_url,
                api_key,
                model,
                messages,
                user_input,
            )
            console.print()

