## AI Advent Challenge 2025

A hands-on “day-by-day” repository of small, runnable AI/LLM demos: prompt shaping, interaction patterns, temperature/tokens, sub-agents, dialogue compression, MCP tool integration, external memory, voice input, document indexing + RAG, reranking/filtering, citations, and image generation + vision QA + meeting summarization.

Everything is wired through a single CLI entrypoint: `main.py`.

### Quickstart

- **Requirements**: Python **3.10+**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# set at least OPENAI_API_KEY (and optionally others)
python main.py --day 1
```

### Configuration (.env)

The code loads environment variables via `python-dotenv` (see `utils/config.py`). Common variables:

- **OpenAI**
  - `OPENAI_API_KEY` (required for days that call OpenAI HTTP APIs)
  - `OPENAI_MODEL` (default: `gpt-4o-mini`)
  - `OPENAI_BASE_URL` (default: `https://api.openai.com`)
  - `OPENAI_VISION_MODEL` (default: `gpt-4o`)
  - `OPENAI_AUDIO_MODEL` (default: `gpt-4o-mini-transcribe`)
- **fal.ai (image generation / vision QA)**
  - `FAL_API_KEY` (required for days 18–19 and for using `day17_image_generation/image_generator.py`)
  - `FAL_VISION_MODEL` (optional; default used in code is `fal-ai/sa2va/4b/image`)
- **MCP (Day 8)**
  - `MCP_SERVER_COMMAND` (stdio transport; e.g. `python path/to/server.py`)
  - `MCP_SERVER_URL` (HTTP transport; e.g. `http://localhost:8000/mcp`)

### How to run

The CLI supports running by **day number** or **task name**:

```bash
python main.py                # defaults to Day 1
python main.py --day 4
python main.py --task rag      # aliases exist (see below)
```

- **Passing arguments through to a task**: anything after the selected task is forwarded to that day’s `run(...)`.

Example (Day 20 takes a required file path argument):

```bash
python main.py --day 20 path/to/meeting.m4a --brief --output outputs/meeting_summary.md
```

### Project structure

- `main.py`: CLI router that imports and runs `dayXX_*/task.py::run`.
- `utils/`: shared config + Rich helpers.
- `day01_simple_ai_agent/` … `day20_meeting_summarizer/`: one folder per day.
- `outputs/`: created by some tasks to store indexes, logs, and generated artifacts.

### Days overview

#### Day 1 — Simple AI agent (terminal chat)
- **What it does**: Minimal terminal chat agent using HTTP `POST /v1/chat/completions` and Rich markdown rendering.
- **Run**: `python main.py --day 1`

#### Day 2 — AI configuration via prompting (strict JSON)
- **What it does**: Demonstrates “structured output” by forcing the model to return only JSON.
- **Run**: `python main.py --day 2`

#### Day 3 — Interaction design
- **What it does**: A guided, multi-turn “workout plan generator” that asks questions and produces a final markdown document.
- **Run**: `python main.py --day 3`

#### Day 4 — Temperature
- **What it does**: Runs a fixed set of prompts at multiple temperatures and compares outputs.
- **Run**: `python main.py --day 4`

#### Day 5 — Tokens
- **What it does**: Uses `tiktoken` to estimate tokens, compares with API `usage`, and demonstrates context-limit behavior.
- **Run**: `python main.py --day 5`

#### Day 6 — Sub-agents interaction
- **What it does**: Two-agent pipeline: one generates code (higher temperature), another validates/improves (temperature 0).
- **Run**: `python main.py --day 6`

#### Day 7 — Dialogue compression
- **What it does**: Runs a long scripted conversation with/without periodic summarization to save tokens, then compares quality.
- **Run**: `python main.py --day 7`

#### Day 8 — MCP integration
- **What it does**: Connects to an MCP server (stdio or HTTP) and lists available tools.
- **Run**: `python main.py --day 8`
- **Requires**: `MCP_SERVER_COMMAND` or `MCP_SERVER_URL`.

#### Day 9 — Custom MCP tool
- **What it does**: Starts a local MCP server (`day09_custom_mcp_tool/server.py`) exposing a real `get_weather` tool (Open-Meteo), then lets the LLM call it.
- **Run**: `python main.py --day 9`

#### Day 10 — MCP tools composition
- **What it does**: Demonstrates multi-step tool chaining with an MCP server that provides `searchDocs` → `summarize` → `saveToFile`.
- **Run**: `python main.py --day 10`
- **Output**: writes files under `outputs/`.

#### Day 11 — External memory
- **What it does**: Persists conversation history to a local SQLite DB and reloads it on next run.
- **Run**: `python main.py --day 11`
- **Clear memory**: `python main.py --day 11 --clear-memory` (or `-c`)

#### Day 12 — Voice agent
- **What it does**: Speech → text (SpeechRecognition/Google) → LLM → response in terminal; can fall back to text mode.
- **Run**: `python main.py --day 12`
- **System dependency**: `pyaudio` often requires **PortAudio**.

#### Day 13 — Document indexing
- **What it does**: Builds a local FAISS index from a directory by chunking files and embedding chunks (`text-embedding-3-small`).
- **Run**: `python main.py --day 13`
- **Output**: `outputs/document_index.faiss` and `outputs/document_index_metadata.json`

#### Day 14 — RAG query
- **What it does**: Compares answers with and without retrieval-augmented context from the Day 13 index.
- **Run**: `python main.py --day 14`
- **Requires**: run Day 13 first.

#### Day 15 — Reranking & filtering
- **What it does**: Adds an LLM-based reranker (“cross-encoder scoring”) and threshold filtering on retrieved chunks.
- **Run**: `python main.py --day 15`
- **Requires**: run Day 13 first.

#### Day 16 — Citations & sources
- **What it does**: Forces citation markers (`[1]`, `[2]`, …) in answers and validates citation correctness.
- **Run**: `python main.py --day 16`
- **Requires**: run Day 13 first.

#### Day 17 — Image generation utilities (fal.ai)
- **What it is**: `day17_image_generation/image_generator.py` provides a reusable `generate_image(...)` helper (fal.ai + optional cost estimate).
- **Note**: Day 17’s `task.py` is currently empty, so the CLI day runner does not have a runnable `run()` entrypoint for Day 17.

#### Day 18 — Prompt & style systems (images)
- **What it does**: Loads JSON style profiles, builds controlled prompts, generates images per style, and analyzes consistency/distinctiveness.
- **Run**: `python main.py --day 18`
- **Requires**: `FAL_API_KEY`
- **Output**: `outputs/day18_generations.json`, `outputs/day18_results.md`

#### Day 19 — Vision QA agent
- **What it does**: Generates images, then evaluates them with a fal.ai vision model against a style checklist; filters by a quality threshold.
- **Run**: `python main.py --day 19`
- **Requires**: `FAL_API_KEY`
- **Output**: `outputs/day19_qa_results.json`

#### Day 20 — Meeting notes summarizer
- **What it does**: Transcribes an audio/video file via OpenAI audio transcription API, then summarizes into structured meeting notes.
- **Run**:

```bash
python main.py --day 20 path/to/file.m4a
python main.py --day 20 path/to/file.mp3 --brief --output outputs/summary.md
```

### Task aliases

`main.py` supports running by name (`--task`) with multiple aliases (examples):

- `--task prompting` → Day 1
- `--task temperature` → Day 4
- `--task tokens` → Day 5
- `--task mcp` → Day 8
- `--task rag` → Day 14
- `--task reranking` → Day 15
- `--task citations` → Day 16
- `--task vision_qa` → Day 19
- `--task meeting_summarizer` → Day 20

(See `main.py` for the full alias map.)

### Troubleshooting

- **401 Unauthorized**: verify `OPENAI_API_KEY` (and/or `FAL_API_KEY`) is set.
- **429 / rate limits**: reduce prompt size, retry later, or switch models.
- **FAISS import error**: install `faiss-cpu` (already in `requirements.txt`, but platform wheels can vary).
- **PyAudio install issues**: install PortAudio first.
  - macOS: `brew install portaudio`
  - Ubuntu/Debian: `sudo apt-get install portaudio19-dev`

### Contributing

- Keep changes lightweight and runnable.
- Prefer configuration via environment variables (see `utils/config.py`).
