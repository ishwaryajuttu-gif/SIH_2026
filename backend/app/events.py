"""Timestamped event logging -> activity timeline for experiment traceability.

Every event is kept in memory (for the dashboard) and appended to a JSON-Lines file
under logs/ (one file per session). Only METADATA is stored - no video frames.

Event fields: timestamp, event_type, activity, state, severity, subject (object or zone),
stability (activity stability, when meaningful), description (+ free-form data).
"""
from __future__ import annotations

import csv
import io
import itertools
import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

from .config import resolve_path

log = logging.getLogger("bas.events")

SEVERITIES = ("info", "success", "warning", "critical")
SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}


@dataclass
class Event:
    id: int
    ts: float                 # unix epoch seconds
    time: str                 # HH:MM:SS local
    type: str                 # event_type, e.g. PERSON_DETECTED, RESTRICTED_ZONE_ENTRY
    severity: str             # info | success | warning | critical
    message: str              # human-readable description
    activity: str = ""        # e.g. SAMPLE_CONTAINER_INTERACTION
    state: str = ""           # state of the component after the event, e.g. INTERACTING, WARNING
    subject: str = ""         # object or zone involved
    stability: float | None = None  # activity stability 0..1 (heuristic, not a classifier probability)
    data: dict = field(default_factory=dict)
    acknowledged: bool = False

    @property
    def is_alert(self) -> bool:
        return SEV_RANK[self.severity] >= SEV_RANK["warning"]


def activity_code(label: str, suffix: str) -> str:
    """'Sample Container Stand-in' + 'INTERACTION' -> 'SAMPLE_CONTAINER_INTERACTION'."""
    base = label.replace("Stand-in", "").replace("-", " ").strip().upper().replace(" ", "_")
    return f"{base}_{suffix}" if base else suffix


class EventLog:
    def __init__(self, cfg: dict):
        lcfg = cfg.get("logging", {})
        self.dir = resolve_path(lcfg.get("dir", "logs"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events: deque[Event] = deque(maxlen=int(lcfg.get("keep_in_memory", 1000)))
        self.alert_window_s = float(lcfg.get("active_alert_window_s", 120))
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Event], None]] = []
        self.session_start = time.time()
        self.file = self.dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        self.write_error = ""

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._listeners.append(fn)

    def emit(self, type_: str, severity: str, message: str, *, activity: str = "", state: str = "",
             subject: str = "", stability: float | None = None, **data) -> Event:
        if severity not in SEV_RANK:
            raise ValueError(f"unknown severity {severity}")
        now = time.time()
        ev = Event(
            id=next(self._ids), ts=now, time=datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            type=type_, severity=severity, message=message, activity=activity or type_, state=state,
            subject=subject, stability=None if stability is None else round(float(stability), 3), data=data,
        )
        with self._lock:
            self.events.append(ev)
            try:
                with open(self.file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
                self.write_error = ""
            except OSError as e:
                self.write_error = str(e)
        for fn in self._listeners:
            try:
                fn(ev)
            except Exception as e:  # noqa: BLE001 - a listener (e.g. TTS) must never break logging
                log.warning("event listener failed: %s", e)
        return ev

    # ---------- queries ----------
    def since(self, last_id: int) -> list[Event]:
        with self._lock:
            return [e for e in self.events if e.id > last_id]

    def recent(self, limit: int = 200) -> list[Event]:
        with self._lock:
            return list(self.events)[-limit:]

    def active_alerts(self, now: float | None = None) -> list[Event]:
        cutoff = (now or time.time()) - self.alert_window_s
        with self._lock:
            return [e for e in self.events if e.is_alert and not e.acknowledged and e.ts >= cutoff]

    def acknowledge(self, event_id: int | None = None) -> int:
        n = 0
        with self._lock:
            for e in self.events:
                if e.is_alert and not e.acknowledged and (event_id is None or e.id == event_id):
                    e.acknowledged = True
                    n += 1
        return n

    def counts(self) -> dict[str, int]:
        with self._lock:
            c = {s: 0 for s in SEVERITIES}
            for e in self.events:
                c[e.severity] += 1
            return c

    CSV_COLUMNS = ["id", "timestamp", "time", "event_type", "activity", "state", "severity",
                   "subject", "stability", "description", "data"]

    def export_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(self.CSV_COLUMNS)
        with self._lock:
            for e in self.events:
                w.writerow([
                    e.id, datetime.fromtimestamp(e.ts).isoformat(timespec="milliseconds"), e.time, e.type,
                    e.activity, e.state, e.severity, e.subject, "" if e.stability is None else e.stability,
                    e.message, json.dumps(e.data, ensure_ascii=False),
                ])
        return buf.getvalue()

    def export_json(self) -> list[dict]:
        with self._lock:
            out = []
            for e in self.events:
                d = asdict(e)
                d["timestamp"] = datetime.fromtimestamp(e.ts).isoformat(timespec="milliseconds")
                d["event_type"] = d.pop("type")
                d["description"] = d.pop("message")
                out.append(d)
            return out
