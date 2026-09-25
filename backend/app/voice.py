"""Offline voice alerts (no internet). pyttsx3 -> SAPI5 on Windows, eSpeak on Linux.

Runs in one dedicated thread (the engine is created and used only there, as SAPI/COM
requires). If TTS cannot initialise, `available` stays False, `error` explains why, and the
rest of the pipeline - including visual alerts - keeps working.
"""
from __future__ import annotations

import logging
import queue
import threading
import time

from .events import Event

log = logging.getLogger("bas.voice")

# event type -> phrase key in config voice.phrases
PHRASE_BY_TYPE = {
    "RESTRICTED_ZONE_ENTRY": "restricted_zone",
    "RESTRICTED_ZONE_CRITICAL": "critical",
    "RESTRICTED_OBJECT": "critical",
    "UNEXPECTED_MOVEMENT": "unexpected",
    "OUT_OF_SEQUENCE": "unexpected",
    "OBJECT_MISSING": "unexpected",
    "CAMERA_SHIFT": "unexpected",
    "PROLONGED_INTERACTION": "unexpected",
}
WORKFLOW_TYPES = {"WORKFLOW_STEP_COMPLETED", "WORKFLOW_COMPLETED"}


class VoiceAlerter:
    def __init__(self, cfg: dict, start_thread: bool = True):
        v = cfg["voice"]
        self.enabled = bool(v["enabled"])
        self.rate = int(v["rate"])
        self.cooldown_s = float(v["cooldown_s"])
        self.speak_workflow = bool(v["speak_workflow"])
        self.phrases: dict[str, str] = dict(v["phrases"])
        self.available = False
        self.state = "starting"     # starting | ready | error
        self.error = ""
        self.spoken = 0
        self.last_spoken = ""
        self._q: queue.Queue[str] = queue.Queue(maxsize=3)
        self._last: dict[str, float] = {}
        self._ready = threading.Event()
        if start_thread:
            threading.Thread(target=self._run, name="voice", daemon=True).start()

    def _run(self):
        com = None
        try:
            try:  # SAPI needs COM initialised in THIS thread on Windows
                import pythoncom  # type: ignore
                pythoncom.CoInitialize()
                com = pythoncom
            except Exception:  # noqa: BLE001 - not Windows / pywin32 missing: pyttsx3 handles it
                pass
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception as e:  # noqa: BLE001
            self.state, self.error = "error", f"Offline TTS unavailable: {e}"
            log.warning(self.error)
            self._ready.set()
            return
        self.available, self.state = True, "ready"
        self._ready.set()
        while True:
            text = self._q.get()
            try:
                engine.say(text)
                engine.runAndWait()
                self.spoken += 1
                self.last_spoken = text
            except Exception as e:  # noqa: BLE001
                self.error = f"TTS failed: {e}"
                log.warning(self.error)
                time.sleep(0.5)
        if com:  # pragma: no cover
            com.CoUninitialize()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    def say(self, text: str, key: str | None = None, now: float | None = None) -> bool:
        """Queue a phrase unless muted or the same key was spoken within the cooldown."""
        if not self.enabled:
            return False
        now = now or time.time()
        k = key or text
        if now - self._last.get(k, -1e9) < self.cooldown_s:
            return False
        self._last[k] = now
        try:
            self._q.put_nowait(text)
            return True
        except queue.Full:
            return False

    def phrase_for(self, ev: Event) -> str | None:
        key = PHRASE_BY_TYPE.get(ev.type)
        if key:
            return self.phrases.get(key)
        if self.speak_workflow and ev.type in WORKFLOW_TYPES:
            return ev.message.split(":")[0]
        return None

    def on_event(self, ev: Event):
        text = self.phrase_for(ev)
        if text:
            # cooldown per phrase: the same sentence is never repeated back-to-back
            self.say(text, key=text, now=ev.ts)
