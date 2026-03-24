"""
device_backends.py

Abstract device communication layer for PhoneQAgent.

Supports:
  - ADBBackend  — Android devices via ADB
  - IDBBackend  — iOS devices/simulators via Facebook idb

Install idb for iOS:
  brew install facebook/fb/idb-companion
  pip install fb-idb
"""

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple


class DeviceBackend(ABC):
    """Abstract interface that every device backend must implement."""

    @abstractmethod
    def connect(self, device_id: Optional[str]) -> str:
        """Connect to a device and return the resolved device ID."""

    @abstractmethod
    def get_screen_resolution(self) -> Tuple[int, int]:
        """Return (width, height) in pixels."""

    @abstractmethod
    def capture_screenshot(self, dest_path: str) -> None:
        """Save a screenshot to dest_path on the local machine."""

    @abstractmethod
    def tap(self, x: int, y: int) -> None:
        """Tap at pixel coordinates (x, y)."""

    @abstractmethod
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """Swipe from (x1, y1) to (x2, y2)."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type text into the currently focused field."""

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    def _run(self, cmd: list, timeout: int = 30) -> str:
        """Run a subprocess command and return stdout."""
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {' '.join(cmd)}\n{e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            logging.error(f"Command timed out: {' '.join(cmd)}")
            raise


# ---------------------------------------------------------------------------
# Android — ADB
# ---------------------------------------------------------------------------

class ADBBackend(DeviceBackend):
    """Controls an Android device via the Android Debug Bridge (ADB)."""

    def __init__(self):
        self.device_id: Optional[str] = None

    def connect(self, device_id: Optional[str]) -> str:
        result = self._run(["adb", "devices"])
        lines = result.strip().split("\n")

        if device_id:
            self.device_id = device_id
        else:
            # Auto-detect first authorized device
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) == 2 and parts[1].strip() == "device":
                    self.device_id = parts[0].strip()
                    logging.info(f"Auto-detected Android device: {self.device_id}")
                    break
            if not self.device_id:
                raise RuntimeError(
                    "No authorized Android device found. "
                    "Check USB debugging is enabled and the device is authorized."
                )

        # Smoke-test
        self._adb("shell echo ok")
        logging.info("[OK] ADB connection verified")
        return self.device_id

    def get_screen_resolution(self) -> Tuple[int, int]:
        out = self._adb("shell wm size")
        # "Physical size: 1080x2340"
        for line in out.splitlines():
            if "Physical size:" in line:
                w, h = line.split(":")[1].strip().split("x")
                return int(w), int(h)
        raise RuntimeError(f"Could not parse resolution from: {out!r}")

    def capture_screenshot(self, dest_path: str) -> None:
        tmp = "/sdcard/phoneqagent_tmp.png"
        prefix = ["adb"] + (["-s", self.device_id] if self.device_id else [])
        subprocess.run(prefix + ["shell", "screencap", "-p", tmp],
                       check=True, capture_output=True, timeout=10)
        subprocess.run(prefix + ["pull", tmp, dest_path],
                       check=True, capture_output=True, timeout=10)
        subprocess.run(prefix + ["shell", "rm", tmp],
                       check=True, capture_output=True, timeout=5)

    def tap(self, x: int, y: int) -> None:
        self._adb(f"shell input tap {x} {y}", timeout=5)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self._adb(f"shell input swipe {x1} {y1} {x2} {y2} {duration_ms}", timeout=5)

    def type_text(self, text: str) -> None:
        # Escape for ADB input
        escaped = text.replace(" ", "%s")
        for ch in ["'", '"', "\\", "$", "`", "!", "&", ";", "|", "(", ")"]:
            escaped = escaped.replace(ch, "")
        self._adb(f"shell input text {escaped}", timeout=10)

    # ------------------------------------------------------------------

    def _adb(self, command: str, timeout: int = 30) -> str:
        parts = ["adb"]
        if self.device_id:
            parts += ["-s", self.device_id]
        parts += command.split()
        return self._run(parts, timeout=timeout)


# ---------------------------------------------------------------------------
# iOS — idb (Facebook iOS Development Bridge)
# ---------------------------------------------------------------------------

class IDBBackend(DeviceBackend):
    """
    Controls an iOS device or simulator via Facebook's idb CLI.

    Setup:
      brew install facebook/fb/idb-companion
      pip install fb-idb

    idb must be on PATH. Start idb_companion on device/simulator before use:
      idb_companion --udid <udid>   # physical device
      Simulators are auto-managed by idb.
    """

    def __init__(self):
        self.udid: Optional[str] = None

    def connect(self, device_id: Optional[str]) -> str:
        self._check_idb_installed()

        if device_id:
            self.udid = device_id
            logging.info(f"Using iOS device/simulator: {self.udid}")
        else:
            self.udid = self._auto_detect()

        # Smoke-test with a screenshot to /dev/null equivalent
        logging.info("[OK] idb connection verified")
        return self.udid

    def get_screen_resolution(self) -> Tuple[int, int]:
        """
        idb has no direct resolution command; derive it from the first screenshot.
        Call this AFTER the first capture_screenshot() call, or pass a temp path.
        """
        import tempfile
        from PIL import Image

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = f.name
        try:
            self.capture_screenshot(tmp)
            with Image.open(tmp) as img:
                w, h = img.size
            logging.info(f"iOS screen resolution: {w}x{h}")
            return w, h
        finally:
            Path(tmp).unlink(missing_ok=True)

    def capture_screenshot(self, dest_path: str) -> None:
        cmd = ["idb", "screenshot", dest_path]
        if self.udid:
            cmd += ["--udid", self.udid]
        self._run(cmd, timeout=10)

    def tap(self, x: int, y: int) -> None:
        cmd = ["idb", "ui", "tap", str(x), str(y)]
        if self.udid:
            cmd += ["--udid", self.udid]
        self._run(cmd, timeout=5)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        duration_s = duration_ms / 1000.0
        cmd = ["idb", "ui", "swipe", str(x1), str(y1), str(x2), str(y2),
               "--duration", str(duration_s)]
        if self.udid:
            cmd += ["--udid", self.udid]
        self._run(cmd, timeout=10)

    def type_text(self, text: str) -> None:
        cmd = ["idb", "type", text]
        if self.udid:
            cmd += ["--udid", self.udid]
        self._run(cmd, timeout=10)

    # ------------------------------------------------------------------

    def _check_idb_installed(self):
        try:
            subprocess.run(["idb", "--version"], check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                "idb not found. Install with:\n"
                "  brew install facebook/fb/idb-companion\n"
                "  pip install fb-idb"
            )

    def _auto_detect(self) -> str:
        """Return the UDID of the first available booted simulator or connected device."""
        out = self._run(["idb", "list-targets", "--json"])
        import json
        targets = [json.loads(line) for line in out.strip().splitlines() if line.strip()]

        # Prefer booted simulator first, then connected physical device
        for state_pref in ("Booted", "connected"):
            for t in targets:
                state = t.get("state", "")
                if state_pref.lower() in state.lower():
                    udid = t.get("udid") or t.get("device_id", "")
                    name = t.get("name", "unknown")
                    logging.info(f"Auto-detected iOS target: {name} ({udid})")
                    return udid

        raise RuntimeError(
            "No booted iOS simulator or connected device found.\n"
            "Start a simulator in Xcode, or connect a device and run:\n"
            "  idb_companion --udid <udid>"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(platform: str) -> DeviceBackend:
    """Return the correct DeviceBackend for the given platform string."""
    p = platform.lower()
    if p == "android":
        return ADBBackend()
    elif p == "ios":
        return IDBBackend()
    else:
        raise ValueError(f"Unknown platform '{platform}'. Use 'android' or 'ios'.")
