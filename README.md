# PhoneQAgent — Android/iOS Automation with Vision AI

![alt text](firefox_wuZjocmz0X.png)

## Overview

**PhoneQAgent** is a Python-based Android/iOS automation agent that uses vision-language models to visually understand and control mobile devices via ADB (Android) or idb (iOS). Describe a task in plain English — the agent captures screenshots, analyzes them with AI, and executes the right actions autonomously.

Supports two backends:
- **Local model** (Qwen3-VL, requires GPU) — runs fully offline
- **Claude API** (Anthropic) — no GPU needed, cloud-based inference

---

## Features

### 🤖 AI-Powered Automation
- **Vision-Language Understanding**: Qwen3-VL (local, 4B/8B/30B) or Claude API (cloud)
- **Natural Language Tasks**: Describe tasks in plain English — no scripting required
- **Context-Aware Actions**: Maintains action history and learns from previous interactions

### 🎮 Control Capabilities
- **Tap** — Precise coordinate-based clicking
- **Swipe** — Directional scrolling for navigation
- **Type** — Text input with automatic field detection
- **Wait** — Smart delays for app loading and transitions
- **Terminate** — Graceful task completion with success/failure status

### 🛡️ Reliability & Error Handling
- **Loop Detection**: Tracks repeated actions and screen hashes; warns the model when stuck
- **Screen Change Monitoring**: MD5 hash comparison detects when UI hasn't updated
- **Failure Tracking**: Logs failed attempts and shares them with the model to prevent retries
- **Task Verification**: Final completion check after max cycles using vision model

### 🖥️ Interfaces
- **Web UI** (Gradio): Live device preview, real-time logs, settings editor
- **CLI**: `python phone_agent.py "your task"`

---

## Platform Support

| Platform | Tool | Physical device | Simulator |
|----------|------|-----------------|-----------|
| Android  | ADB  | ✅ | ✅ (emulator) |
| iOS      | idb  | ✅ | ✅ (Xcode Simulator) |

Set `"platform": "android"` or `"platform": "ios"` in `config.json`.

---

## Backend Options

### Option A: Claude API (No GPU required)

Uses the Anthropic Claude API — no local model download, no GPU needed.

**Install:**
```bash
pip install anthropic
```

**Configure `config.json`:**
```json
{
  "use_claude_api": true,
  "claude_model": "claude-opus-4-6",
  "anthropic_api_key": "sk-ant-..."
}
```

Or set the environment variable instead of putting the key in config:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then run as normal:
```bash
python ui.py
# or
python phone_agent.py "Open Chrome and search for weather"
```

### Option B: Local Qwen3-VL Model (GPU required)

Runs fully offline. Requires a CUDA GPU with sufficient VRAM.

| Model | VRAM | Notes |
|-------|------|-------|
| Qwen3-VL-8B-Instruct | ~11 GB | **Recommended** — best performance/speed balance |
| Qwen3-VL-4B-Instruct | ~6 GB | Lightweight option |
| Qwen3-VL-30B-A3B-Instruct (MoE) | 128 GB+ | Not recommended — 8B dense is 3–5x faster |

Set `"use_claude_api": false` in `config.json` (default) and set `"model_name"` to your chosen model.