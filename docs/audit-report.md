# Engineering audit report (v0.1 → v0.2)

This audit covers the complete v0.1 repository: backend modules, frontend, config, scripts, tests and docs.

## Implemented vs documented-only (v0.1)

| Claimed in v0.1 docs | Reality in v0.1 code | v0.2 action |
|---|---|---|
| YOLO + MediaPipe + interaction + state machine + safety + logging + voice + dashboard | Implemented and working on the sample video | Kept; hardened |
| "Runs on a Jetson (JetPack) unchanged" | Never tested | Claim removed; replaced with accurate edge wording |
| "~8–15 FPS on a laptop", "OpenVINO ~2x faster" | Not measured on a laptop | Replaced with the measured cloud figure plus a note that the laptop is unmeasured |
| Equipment labels "Sample Container", "Crew Member", experiment "Plant Growth Payload" | Pretrained COCO classes, no BAS data | Renamed "… Stand-in", "Person"; honest app title |
| "Sample Transfer Protocol" | Invented demo sequence | Renamed "Representative Demo Workflow", with a disclaimer |
| "Recognition confidence" | `det_conf × (0.6 + 0.4·stability) + 0.15`: an arbitrary +15% boost | Replaced by **Activity Stability** (temporal consistency), with detection confidence shown separately |
| System status "green" for YOLO / HOI | Green whenever *any* state existed | Health is now evidence-based (loaded, ran recently, frames arriving), with ERROR reasons |
| YOLO-World auto-labelling | Code present; never run end to end (model download blocked) | Still untested; documented as optional future tooling |

## Defects found and fixed

| # | Defect | Fix |
|---|---|---|
| 1 | Hand-tracking loss during contact counted as a release, giving a **false COMPLETED** after about 1 s | Hand-loss grace period. The state is held (HAND TRACKING LOST); afterwards the interaction is "interrupted", never completed. |
| 2 | An object track dropped mid-interaction was **counted as a completed "move"** and could satisfy a workflow step | Logged as OBJECT_LOST_DURING_INTERACTION; not counted |
| 3 | Missing YOLO weights triggered an **internet download at run time** | Local weights required; clear ERROR otherwise |
| 4 | A missing hand model failed silently to backend "none" | Clear ERROR with the fix shown in the dashboard |
| 5 | `run.bat` **opened the browser before starting the server** | `launch.py`: start the server, poll `/api/health`, then open the browser |
| 6 | Server bound to **0.0.0.0** (webcam stream exposed on the LAN), CORS `*` | Binds to 127.0.0.1; CORS limited to the dev server |
| 7 | Camera "ok" stayed true after frames stopped; no retries; no camera discovery; DSHOW only | Frame-verified open with DSHOW, then MSMF, then any other back-end; stall detection; reconnect; `probe_cameras`; selector in the UI |
| 8 | Restricted zone went **straight to CRITICAL**; the zone-exit threshold was hard-coded; no enable/disable or zone types | WARNING, then CRITICAL escalation; time-based exit debounce; enable toggle; workstation zone type; rectangle/polygon editor |
| 9 | Unexpected-movement rule fired when **hands simply weren't detected** (normal pick-up with a lost hand) | Rule requires hands tracked in at least 60% of the window (or nobody present) |
| 10 | API threads (source switch, zone update, reset) **raced** with the pipeline thread | A single lock around processing and state changes |
| 11 | "Object detected" announcement could fire twice with `detect_every > 1` | Per-object flag |
| 12 | Voice reported "available" even if SAPI failed; phrases were cut from messages; no COM init in the thread | Engine initialised once in its own thread with COM init; explicit state and error; fixed phrases with a cooldown |
| 13 | IoU-only tracker lost identity on fast moves | Same-label centre-distance fallback (tracker kept simple) |
| 14 | Hard-coded thresholds (presence frames, stable-object frames, zone exit); no config validation | All in `config.yaml`, validated at start-up; a test checks that every key is used |
| 15 | Events lacked activity/state/subject/stability fields | Added to events, CSV and JSON |
| 16 | JPEG encoded every frame even with no viewer | Encoded only while the stream is watched |
| 17 | Zone save stripped comments from `config.yaml` (fixed during v0.1) | Zones stored in `data/zones.json`, now validated |
| 18 | `download_models.py` could leave a half-downloaded model that counted as "present" | Downloads to a `.part` file, validated, then renamed |

## Kept (not replaced)

These were adequate and were kept:

- YOLO / MediaPipe integration
- the per-object state-machine design
- the safety-engine structure
- event logging
- voice through pyttsx3
- the IoU tracker
- the FastAPI routes (renamed `/api/procedure` to `/api/workflow`; `/api/source` now takes `{kind, value}`)
- the React dashboard layout
- all original test scenarios, adapted where behaviour intentionally changed; none deleted
