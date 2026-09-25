"""Representative demo workflow engine (expected-sequence validation).

This workflow is a representative demonstration sequence created to validate the
prototype's activity-recognition architecture. It is NOT an official BAS/ISRO protocol.
Completed activities from the state machine are compared with the expected step order;
steps done early are flagged as out of sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

from .activity import CompletedActivity
from .events import EventLog, activity_code


@dataclass
class Step:
    id: str
    name: str
    object: str
    action: str                 # interact | move
    status: str = "pending"     # pending | active | done
    completed_at: float | None = None

    def matches(self, act: CompletedActivity) -> bool:
        if act.object == self.object:
            return self.action != "move" or act.action == "move"
        # Support both custom model labels (e.g. "Sample Container") and
        # stand-in labels (e.g. "Sample Container Stand-in")
        norm_step = self.object.replace(" Stand-in", "").strip().lower()
        norm_act = act.object.replace(" Stand-in", "").strip().lower()
        if norm_step == norm_act:
            return self.action != "move" or act.action == "move"
        return False


class WorkflowTracker:
    def __init__(self, cfg: dict, events: EventLog):
        w = cfg["workflow"]
        self.name = w.get("name", "Representative Demo Workflow")
        self._raw_steps = w.get("steps", []) or []
        self.events = events
        self.reset(announce=False)

    def reset(self, announce: bool = True):
        self.steps = [Step(s["id"], s["name"], s["object"], s.get("action", "interact")) for s in self._raw_steps]
        self.index = 0
        self.completed = False
        self.started_at: float | None = None
        self.deviations = 0
        if self.steps:
            self.steps[0].status = "active"
        if announce:
            self.events.emit("WORKFLOW_RESET", "info", f"'{self.name}' reset", activity="WORKFLOW_RESET",
                             state="STEP_1")

    @property
    def tracked_objects(self) -> set[str]:
        objs = {s.object for s in self.steps}
        equiv = set()
        for o in objs:
            if " Stand-in" in o:
                equiv.add(o.replace(" Stand-in", "").strip())
            else:
                equiv.add(f"{o} Stand-in")
        return objs | equiv

    def on_activity(self, act: CompletedActivity):
        if self.completed or not self.steps or act.object not in self.tracked_objects:
            return
        cur = self.steps[self.index]
        if cur.matches(act):
            if self.started_at is None:
                self.started_at = act.ts
            cur.status, cur.completed_at = "done", act.ts
            self.events.emit(
                "WORKFLOW_STEP_COMPLETED", "success", f"Demo workflow step {cur.id} completed: {cur.name}",
                activity=activity_code(cur.object, "STEP_DONE"), state=f"STEP_{cur.id}_DONE", subject=cur.object,
                step=cur.id,
            )
            self.index += 1
            if self.index >= len(self.steps):
                self.completed = True
                total = act.ts - (self.started_at or act.ts)
                self.events.emit(
                    "WORKFLOW_COMPLETED", "success",
                    f"Demo workflow completed ({len(self.steps)} steps, {self.deviations} deviations, {total:.0f}s)",
                    activity="WORKFLOW_COMPLETED", state="COMPLETED", deviations=self.deviations,
                )
            else:
                self.steps[self.index].status = "active"
            return

        for s in self.steps[: self.index]:
            if s.matches(act):
                self.events.emit("WORKFLOW_STEP_REPEATED", "info", f"Step {s.id} repeated: {s.name}",
                                 activity=activity_code(s.object, "STEP_REPEATED"), subject=s.object, step=s.id)
                return

        for s in self.steps[self.index + 1:]:
            if s.matches(act):
                self.deviations += 1
                self.events.emit(
                    "OUT_OF_SEQUENCE", "warning",
                    f"Out of sequence: expected '{cur.name}' but detected '{s.name}'",
                    activity="OUT_OF_SEQUENCE_ACTIVITY", state="WARNING", subject=s.object,
                    expected=cur.id, detected=s.id,
                )
                return

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "note": "Representative demonstration sequence - not an official BAS/ISRO protocol.",
            "current_index": self.index,
            "completed": self.completed,
            "deviations": self.deviations,
            "steps": [
                {"id": s.id, "name": s.name, "object": s.object, "action": s.action,
                 "status": s.status, "completed_at": s.completed_at}
                for s in self.steps
            ],
        }


ProcedureTracker = WorkflowTracker  # backwards-compatible name
