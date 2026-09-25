"""Configuration loading and validation.

All paths in config.yaml are relative to the backend folder. Every key is declared in
SCHEMA below: unknown keys, wrong types and out-of-range values are reported at start-up,
so there are no silent dead settings. Environment overrides for the launcher:
    BAS_CAMERA=<index>   -> video.mode=webcam, video.camera_index=<index>
    BAS_VIDEO=<path>     -> video.mode=file,   video.file=<path>
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BACKEND_DIR / "config.yaml"


def zones_file() -> Path:
    """Where dashboard-edited zones are stored (tests point BAS_ZONES_FILE at a temp file)."""
    return Path(os.environ.get("BAS_ZONES_FILE") or BACKEND_DIR / "data" / "zones.json")


class ConfigError(ValueError):
    pass


# (type, min, max) - None = unbounded. "map"/"list" are free-form containers checked by type only.
NUM = (int, float)
SCHEMA: dict[str, dict[str, tuple]] = {
    "app": {"title": (str,), "subtitle": (str,)},
    "video": {
        "mode": (str,), "camera_index": (int, 0, 16), "file": (str,),
        "frame_width": (int, 160, 3840), "frame_height": (int, 120, 2160),
        "process_width": (int, 160, 1920), "max_processing_fps": (NUM, 1, 60),
        "loop_video": (bool,), "jpeg_quality": (int, 20, 100),
        "camera_open_timeout_s": (NUM, 1, 60), "camera_stall_s": (NUM, 0.5, 60),
    },
    "detector": {
        "mode": (str,), "model": (str,), "model_path": (str,), "world_model": (str,), "custom_model": (str,),
        "conf": (NUM, 0.01, 0.99), "class_conf": (dict,), "imgsz": (int, 160, 1280),
        "detect_every": (int, 1, 10), "class_map": (dict,), "world_prompts": (dict,),
        "restricted_objects": (list,),
    },
    "tracker": {
        "iou_match": (NUM, 0.01, 0.99), "center_match": (NUM, 0, 5), "max_missed": (int, 1, 300),
        "smoothing": (NUM, 0, 0.95), "min_hits": (int, 1, 30),
    },
    "hands": {
        "enabled": (bool,), "model": (str,), "max_hands": (int, 1, 4),
        "min_detection_conf": (NUM, 0.05, 0.99), "min_tracking_conf": (NUM, 0.05, 0.99),
    },
    "interaction": {"contact_margin": (NUM, 0, 1), "near_factor": (NUM, 0.1, 10)},
    "activity": {
        "approach_confirm_frames": (int, 1, 100), "interaction_confirm_frames": (int, 2, 100),
        "release_confirm_frames": (int, 1, 200), "min_interaction_s": (NUM, 0, 30),
        "interaction_cooldown_s": (NUM, 0, 60), "move_threshold": (NUM, 0.005, 1),
        "missing_as_held_frames": (int, 1, 200), "hand_lost_grace_s": (NUM, 0, 30),
        "max_interaction_s": (NUM, 1, 3600), "person_confirm_frames": (int, 1, 100),
        "person_lost_frames": (int, 1, 1000), "stability_window": (int, 3, 200),
    },
    "safety": {
        "restricted_confirm_frames": (int, 1, 100), "zone_exit_s": (NUM, 0, 30),
        "critical_after_s": (NUM, 0, 120), "alert_cooldown_s": (NUM, 0, 600),
        "unexpected_move_threshold": (NUM, 0.01, 1), "unexpected_move_window": (int, 3, 300),
        "stable_object_frames": (int, 1, 1000), "missing_object_s": (NUM, 0.5, 600),
        "recent_alert_s": (NUM, 0, 60), "zones": (list,),
    },
    "workflow": {"name": (str,), "steps": (list,)},
    "voice": {
        "enabled": (bool,), "rate": (int, 60, 400), "cooldown_s": (NUM, 0, 600),
        "speak_workflow": (bool,), "phrases": (dict,),
    },
    "logging": {"dir": (str,), "keep_in_memory": (int, 50, 100000), "active_alert_window_s": (NUM, 5, 86400)},
}
ENUMS = {("video", "mode"): {"webcam", "file"}, ("detector", "mode"): {"coco", "world", "custom"}}
ZONE_TYPES = {"restricted", "workstation"}
VOICE_PHRASES = {"restricted_zone", "unexpected", "critical"}


def resolve_path(p: str | os.PathLike) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    if (BACKEND_DIR / path).exists():
        return BACKEND_DIR / path
    if (BACKEND_DIR.parent / path).exists():
        return BACKEND_DIR.parent / path
    if (BACKEND_DIR.parent / path).parent.exists():
        return BACKEND_DIR.parent / path
    return BACKEND_DIR / path


def validate_zone(z: dict, i: int = 0) -> dict:
    if not isinstance(z, dict):
        raise ConfigError(f"zone #{i + 1} must be a mapping")
    pts = z.get("points")
    if not isinstance(pts, list) or len(pts) < 3:
        raise ConfigError(f"zone '{z.get('name', i + 1)}' needs at least 3 points")
    for p in pts:
        if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, NUM) and 0 <= v <= 1 for v in p)):
            raise ConfigError(f"zone '{z.get('name', i + 1)}' has an invalid point {p} (expected [x, y] in 0..1)")
    ztype = z.get("type", "restricted")
    if ztype not in ZONE_TYPES:
        raise ConfigError(f"zone '{z.get('name', i + 1)}' type must be one of {sorted(ZONE_TYPES)}")
    return {
        "id": str(z.get("id") or f"zone-{i + 1}"),
        "name": str(z.get("name") or f"Zone {i + 1}"),
        "type": ztype,
        "enabled": bool(z.get("enabled", True)),
        "points": [[float(p[0]), float(p[1])] for p in pts],
    }


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Raise ConfigError for invalid values; return a list of non-fatal warnings."""
    warnings: list[str] = []
    for section, keys in cfg.items():
        if section.startswith("_"):
            continue
        if section not in SCHEMA:
            warnings.append(f"unknown section '{section}' (ignored)")
            continue
        if not isinstance(keys, dict):
            raise ConfigError(f"section '{section}' must be a mapping")
        for key in keys:
            if key not in SCHEMA[section]:
                warnings.append(f"unknown key '{section}.{key}' (ignored)")
    for section, keys in SCHEMA.items():
        if section not in cfg:
            raise ConfigError(f"missing section '{section}'")
        for key, spec in keys.items():
            if key not in cfg[section]:
                raise ConfigError(f"missing key '{section}.{key}'")
            val = cfg[section][key]
            typ = spec[0]
            if (isinstance(val, bool) and typ is not bool) or not isinstance(val, typ):
                raise ConfigError(f"'{section}.{key}' has wrong type ({type(val).__name__})")
            if len(spec) == 3 and not (spec[1] <= val <= spec[2]):
                raise ConfigError(f"'{section}.{key}'={val} out of range [{spec[1]}, {spec[2]}]")
            allowed = ENUMS.get((section, key))
            if allowed and val not in allowed:
                raise ConfigError(f"'{section}.{key}' must be one of {sorted(allowed)}")
    cfg["safety"]["zones"] = [validate_zone(z, i) for i, z in enumerate(cfg["safety"]["zones"])]
    for i, s in enumerate(cfg["workflow"]["steps"]):
        if not isinstance(s, dict) or not {"id", "name", "object"} <= set(s):
            raise ConfigError(f"workflow step #{i + 1} needs id, name and object")
        if s.get("action", "interact") not in ("interact", "move"):
            raise ConfigError(f"workflow step {s['id']} action must be interact or move")
    missing = VOICE_PHRASES - set(cfg["voice"]["phrases"])
    if missing:
        raise ConfigError(f"voice.phrases is missing {sorted(missing)}")
    if cfg["detector"]["class_map"].get("person") != "Person":
        raise ConfigError("detector.class_map.person must be 'Person' (used for presence detection)")
    labels = set(cfg["detector"]["class_map"].values()) | set(cfg["detector"]["world_prompts"].values())
    for obj in cfg["detector"]["restricted_objects"]:
        if obj not in labels:
            warnings.append(f"restricted object '{obj}' is not produced by any class_map/world_prompts entry")
    for s in cfg["workflow"]["steps"]:
        if s["object"] not in labels:
            warnings.append(f"workflow step {s['id']} object '{s['object']}' is not a detector label")
    return warnings


def load_config(path: str | os.PathLike | None = None, use_env: bool = True, use_zone_file: bool = True) -> dict[str, Any]:
    cfg_path = Path(path or os.environ.get("BAS_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_path"] = str(cfg_path)
    zf = zones_file()
    if use_zone_file and zf.exists():  # zones drawn in the dashboard override the defaults
        try:
            cfg.setdefault("safety", {})["zones"] = json.loads(zf.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if use_env:
        cam = os.environ.get("BAS_CAMERA", "").strip()
        vid = os.environ.get("BAS_VIDEO", "").strip()
        if cam.isdigit():
            cfg["video"]["mode"], cfg["video"]["camera_index"] = "webcam", int(cam)
        elif vid:
            cfg["video"]["mode"], cfg["video"]["file"] = "file", vid
    cfg["_warnings"] = validate_config(cfg)
    return cfg


def save_zones(cfg: dict[str, Any], zones: list[dict]) -> list[dict]:
    """Validate and persist zones edited in the dashboard to data/zones.json (config.yaml stays untouched)."""
    clean = [validate_zone(z, i) for i, z in enumerate(zones)]
    zf = zones_file()
    zf.parent.mkdir(parents=True, exist_ok=True)
    zf.write_text(json.dumps(clean, indent=2), encoding="utf-8")
    cfg.setdefault("safety", {})["zones"] = clean
    return clean
