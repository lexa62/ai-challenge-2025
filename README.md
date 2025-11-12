AI Advent Challenge 2025
========================

Welcome to the AI Advent Challenge 2025! This repository now ships with a single, ready-to-run Day 1 terminal agent that accepts your input, calls OpenAI over HTTP, and renders markdown responses in the console.

Project Structure
-----------------

ai-challenge-2025/

- day01_simple_ai_agent/
  - agent.py
  - __init__.py
  - task.py
- utils/
  - config.py
  - helpers.py
- .env.example
- requirements.txt
- main.py
- README.md

Getting Started
---------------

Prerequisites:

- Python 3.10+
- pip

Setup:

1) Create and activate a virtual environment

macOS/Linux:

python3 -m venv .venv
source .venv/bin/activate

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\Activate.ps1

2) Install dependencies

pip install -r requirements.txt

3) Configure environment variables

Copy `.env.example` to `.env` and fill in keys:

OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com

Day Layout
----------

- Current day: `day01_simple_ai_agent/` with `agent.py` and `task.py`.

How to Run
----------

python main.py

- Also supported:

python main.py --day 1

How to Exit
-----------

- Type :q or :quit and press Enter
- Press Enter on an empty line
- Ctrl+C (KeyboardInterrupt) or Ctrl+D (EOF on macOS/Linux)

Notes
-----

- The agent uses plain HTTP via `requests` to call `POST /v1/chat/completions`.
- Responses are rendered as markdown inside the terminal (with code blocks and lists).
- Use `utils/config.py` to load environment variables and defaults.

Troubleshooting
---------------

- If you see 401 errors, ensure `OPENAI_API_KEY` is set and valid.
- If you see 429 or timeouts, wait and retry or simplify the prompt.

Contributing
------------

- Keep changes lightweight and readable.
- Prefer environment variables via `.env` and `utils/config.py`.
