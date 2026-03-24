# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An AI-powered Android automation framework that uses Qwen3-VL (vision-language model) to visually understand phone screens and execute tasks autonomously via ADB (Android Debug Bridge). Users describe tasks in natural language; the agent captures screenshots, analyzes them with a VLM, and executes touch/swipe/type actions in a loop.

## Setup & Running

**Environment (conda recommended):**
```bash
conda create -n android_phone_control_agent
conda activate android_phone_control_agent
python -m pip install git+https://github.com/huggingface/transformers  # Must install from source
python -m pip install huggingface_hub pillow gradio qwen_vl_utils requests accelerate hf_xet
# Linux only: add flash-attn
```

**Run options:**
```bash
python ui.py                          # Web UI at http://localhost:7860 (recommended)
python phone_agent.py "task here"     # CLI mode
```

**Configuration:** Edit `config.json` for device ID, screen resolution, model selection, and AI parameters. To use Claude API instead of a local GPU, set `"use_claude_api": true` and either set `"anthropic_api_key"` in config or export `ANTHROPIC_API_KEY`.

## Platform Support

Set `"platform"` in `config.json`:
- `"android"` — uses `ADBBackend` (default)
- `"ios"` — uses `IDBBackend` via Facebook idb (`brew install facebook/fb/idb-companion` + `pip install fb-idb`); macOS only

## Architecture

The system runs a **screenshot → VLM analysis → action → repeat** loop:

```
config.json → PhoneAgent
                ├── DeviceBackend (ADBBackend or IDBBackend)
                │     ├── connect / get_screen_resolution
                │     ├── capture_screenshot
                │     └── tap / swipe / type_text
                ├── VL Agent (ClaudeAPIAgent or QwenVLAgent)
                │     └── analyze_screenshot → action dict
                └── action loop (execute_task → execute_cycle)
```

**`device_backends.py`**
- `DeviceBackend` abstract base class with `connect`, `get_screen_resolution`, `capture_screenshot`, `tap`, `swipe`, `type_text`
- `ADBBackend` — Android via ADB CLI
- `IDBBackend` — iOS via Facebook idb CLI; resolution is inferred from the first screenshot since idb has no dedicated resolution command
- `create_backend(platform)` factory used by `PhoneAgent.__init__`

**`phone_agent.py` — `PhoneAgent` class**
- Orchestrates the full loop (`execute_task()` → `execute_cycle()`)
- Manages ADB connection, screenshot capture, and action execution
- Tracks session context: `previous_actions`, `failed_attempts`, `screen_hash`, `current_app`
- Detects stuck loops via `_detect_repetitive_behavior()` (compares screen hashes + action history)
- Auto-corrects resolution mismatches with `_verify_screen_resolution()`

**`claude_api_agent.py` — `ClaudeAPIAgent` class**
- Drop-in replacement for `QwenVLAgent` — same `analyze_screenshot()` / `check_task_completion()` interface
- Uses `anthropic` Python SDK with streaming; no GPU required
- Sends screenshots as base64 PNG; uses Claude's native tool calling with `tool_choice: {type: "tool", name: "mobile_use"}` to guarantee an action on every call
- Converts Claude tool output to the same internal action dict that `PhoneAgent` expects

**`qwen_vl_agent.py` — `QwenVLAgent` class**
- Wraps Hugging Face Transformers to load and run Qwen3-VL models
- `analyze_screenshot()` builds a context-aware prompt including action history and failures
- `_parse_action()` extracts `tool_call` JSON from model output (tap/swipe/type/wait/terminate)
- `check_task_completion()` does a final VLM verification pass at max cycles

**`ui.py`**
- Gradio app with three tabs: Control Center (task input + live preview), Settings (device config), Documentation
- Runs agent in background thread; streams logs and screenshots to UI in real time

**`qwen_vl_utils.py`**
- Minimal utility: converts PIL images, file paths, and URLs into the format expected by Qwen3-VL

## Key Details

- **Hardware**: Minimum 24GB VRAM for the 8B model; the 8B dense model outperforms the 30B MoE by 3–5x in practice
- **ADB prerequisite**: Android device must have Developer Mode + USB debugging enabled and be visible via `adb devices`
- **Windows**: Install without `flash-attn`; set `PYTHONIOENCODING=utf-8` for UTF-8 handling
- **No safety guardrails**: The agent executes whatever task is described — no filtering of harmful instructions
