"""
claude_api_agent.py

Drop-in replacement for QwenVLAgent that uses the Anthropic Claude API
instead of a locally-loaded model. No GPU required.

Set ANTHROPIC_API_KEY in your environment before use.
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
from PIL import Image


# The mobile_use tool definition in Claude's native tool format
MOBILE_USE_TOOL = {
    "name": "mobile_use",
    "description": (
        "Use a touchscreen to interact with a mobile device, and take screenshots.\n"
        "* This is an interface to a mobile device with touchscreen. You can perform "
        "actions like clicking, typing, swiping, etc.\n"
        "* Some applications may take time to start or process actions, so you may "
        "need to wait and take successive screenshots to see the results of your actions.\n"
        "* The screen's resolution is 999x999.\n"
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in "
        "the center of the element. Don't click boxes on their edges unless asked."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["click", "swipe", "type", "wait", "terminate"],
                "description": (
                    "The action to perform:\n"
                    "* `click`: Click at coordinate (x, y).\n"
                    "* `swipe`: Swipe from (x, y) to (x2, y2).\n"
                    "* `type`: Input text into the active input box.\n"
                    "* `wait`: Wait for the specified number of seconds.\n"
                    "* `terminate`: End the task and report its completion status."
                ),
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "description": "(x, y) to click or start a swipe. Range: 0–999.",
            },
            "coordinate2": {
                "type": "array",
                "items": {"type": "number"},
                "description": "(x, y) end point for swipe. Range: 0–999.",
            },
            "text": {
                "type": "string",
                "description": "Text to type. Required only for action=type.",
            },
            "time": {
                "type": "number",
                "description": "Seconds to wait. Required only for action=wait.",
            },
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": "Task status. Required only for action=terminate.",
            },
        },
        "required": ["action"],
    },
}


class ClaudeAPIAgent:
    """
    Vision-Language agent using the Anthropic Claude API for mobile GUI automation.
    Implements the same interface as QwenVLAgent so it can be used as a drop-in
    replacement in PhoneAgent.
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = anthropic.Anthropic(api_key=api_key)  # falls back to ANTHROPIC_API_KEY

        self.system_prompt = (
            "You are a mobile automation assistant. You control an Android device "
            "by analyzing screenshots and choosing the best action to complete the user's task.\n\n"
            "Rules:\n"
            "- Always call the mobile_use tool — never reply with plain text.\n"
            "- Be brief: reason in one sentence before each action.\n"
            "- Coordinates are in 999×999 space where (0,0) is top-left.\n"
            "- Click the center of UI elements.\n"
            "- If the task is complete or cannot be completed, call mobile_use with action=terminate."
        )

        logging.info(f"ClaudeAPIAgent initialized (model={model})")

    # ------------------------------------------------------------------
    # Public interface (matches QwenVLAgent)
    # ------------------------------------------------------------------

    def analyze_screenshot(
        self,
        screenshot_path: str,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyze a screenshot and return the next action dict."""
        try:
            image_b64, media_type = self._encode_image(screenshot_path)
            user_content = self._build_user_content(
                image_b64, media_type, user_request, context
            )
            return self._call_api(user_content)
        except Exception as e:
            logging.error(f"ClaudeAPIAgent.analyze_screenshot error: {e}", exc_info=True)
            return None

    def check_task_completion(
        self,
        screenshot_path: str,
        user_request: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Ask the model whether the task is complete."""
        try:
            image_b64, media_type = self._encode_image(screenshot_path)
            n_actions = len(context.get("previous_actions", []))
            query = (
                f"The user task: {user_request}\n\n"
                f"You have completed {n_actions} actions so far.\n\n"
                "Look at the current screen and decide: has the task been completed?\n"
                "Call mobile_use with action=terminate and status='success' if done, "
                "or status='failure' explaining what is missing."
            )
            user_content = [
                {"type": "text", "text": query},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_b64,
                    },
                },
            ]
            action = self._call_api(user_content)
            if action and action.get("action") == "terminate":
                return {
                    "complete": action.get("status") == "success",
                    "reason": action.get("message", ""),
                    "confidence": 0.9 if action.get("status") == "success" else 0.7,
                }
            return {"complete": False, "reason": "Unable to verify", "confidence": 0.0}
        except Exception as e:
            logging.error(f"ClaudeAPIAgent.check_task_completion error: {e}")
            return {"complete": False, "reason": str(e), "confidence": 0.0}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_image(self, path: str):
        """Load, optionally resize, and base64-encode an image."""
        img = Image.open(path)
        if max(img.size) > 1280:
            ratio = 1280 / max(img.size)
            img = img.resize(
                (int(img.width * ratio), int(img.height * ratio)),
                Image.Resampling.LANCZOS,
            )
        # Convert to RGB PNG (handles RGBA, palette, etc.)
        from io import BytesIO
        buf = BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        return base64.standard_b64encode(buf.getvalue()).decode(), "image/png"

    def _build_user_content(
        self,
        image_b64: str,
        media_type: str,
        user_request: str,
        context: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Assemble the user message content list."""
        parts: List[str] = [f"Task: {user_request}"]

        if context:
            previous = context.get("previous_actions", [])
            if previous:
                history_lines = []
                for i, act in enumerate(previous[-5:], 1):
                    desc = f"Step {i}: {act.get('action', '?')}"
                    if act.get("action") == "tap" and "coordinates" in act:
                        cx, cy = act["coordinates"]
                        desc += f" at ({int(cx*999)}, {int(cy*999)})"
                    if act.get("elementName"):
                        desc += f" — {act['elementName']}"
                    history_lines.append(desc)
                parts.append(
                    "\n=== PREVIOUS ACTIONS ===\n"
                    + "\n".join(history_lines)
                    + "\n=== END ==="
                )

            if context.get("repetition_warning"):
                parts.append(
                    "\n⚠️ LOOP DETECTED — your last several actions were identical.\n"
                    "You MUST choose a DIFFERENT action or location.\n"
                    "If no alternative exists, terminate with status='failure'.\n"
                    + context["repetition_warning"]
                )

            if context.get("screen_unchanged"):
                parts.append(
                    "\n⚠️ The screen has NOT changed since your last action. "
                    "Try a completely different approach."
                )

            if context.get("recent_failures"):
                summary = "; ".join(
                    f"{f['action']} failed ({f['reason']})"
                    for f in context["recent_failures"]
                )
                parts.append(f"\n📋 Recent failures to avoid: {summary}")

        return [
            {"type": "text", "text": "\n".join(parts)},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_b64},
            },
        ]

    def _call_api(self, user_content: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call the Claude API with tool use and parse the result."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            tools=[MOBILE_USE_TOOL],
            tool_choice={"type": "tool", "name": "mobile_use"},
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            response = stream.get_final_message()

        # Extract the tool_use block
        for block in response.content:
            if block.type == "tool_use" and block.name == "mobile_use":
                return self._parse_tool_input(block.input)

        logging.error("ClaudeAPIAgent: no mobile_use tool call in response")
        return None

    def _parse_tool_input(self, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Claude tool input to the internal action format used by PhoneAgent."""
        action_type = args.get("action")
        if not action_type:
            logging.error("ClaudeAPIAgent: tool input missing 'action'")
            return None

        action: Dict[str, Any] = {"action": action_type}

        # Normalize click → tap (PhoneAgent internal name)
        if action_type == "click":
            action["action"] = "tap"

        # Coordinates — convert from 999×999 space to normalized 0–1
        if "coordinate" in args:
            cx, cy = args["coordinate"]
            action["coordinates"] = [cx / 999.0, cy / 999.0]

        if "coordinate2" in args:
            cx2, cy2 = args["coordinate2"]
            action["coordinate2"] = [cx2 / 999.0, cy2 / 999.0]

        # Derive swipe direction for PhoneAgent compatibility
        if action["action"] == "swipe" and "coordinates" in action and "coordinate2" in action:
            sx, sy = action["coordinates"]
            ex, ey = action["coordinate2"]
            dx, dy = ex - sx, ey - sy
            if abs(dy) > abs(dx):
                action["direction"] = "down" if dy > 0 else "up"
            else:
                action["direction"] = "right" if dx > 0 else "left"

        if "text" in args:
            action["text"] = args["text"]

        if "time" in args:
            action["waitTime"] = int(float(args["time"]) * 1000)

        if "status" in args:
            action["status"] = args["status"]
            action["message"] = f"Task {args['status']}"

        # Validate required fields
        if action["action"] == "tap" and "coordinates" not in action:
            logging.error("ClaudeAPIAgent: tap missing coordinates")
            return None
        if action["action"] == "type" and "text" not in action:
            logging.error("ClaudeAPIAgent: type missing text")
            return None

        return action
