# qwen_vl_utils.py
from typing import Any, Dict, List, Tuple
from PIL import Image


def _as_image(x):
    """
    Accepts PIL.Image, local path, or http(s)/data URL.
    Returns a PIL.Image (RGB) or passes URLs through (processor can fetch).
    """
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    if isinstance(x, str):
        if x.lower().startswith(("http://", "https://", "data:")):
            return x
        return Image.open(x).convert("RGB")
    # Let the processor handle anything else it supports
    return x


def process_vision_info(messages: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    """
    Minimal shim matching the signature that qwen_vl_agent expects.
    Collects image/video inputs from the chat-format messages.
    """
    images, videos = [], []
    for m in messages:
        for c in m.get("content", []):
            t = c.get("type")
            if t == "image":
                images.append(_as_image(c.get("image")))
            elif t == "video":
                videos.append(c.get("video"))
    return images, videos
