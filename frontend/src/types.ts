export type Severity = 'info' | 'success' | 'warning' | 'critical'
export type SceneState =
  | 'IDLE'
  | 'PERSON_DETECTED'
  | 'APPROACHING'
  | 'HAND_NEAR_EQUIPMENT'
  | 'INTERACTING'
  | 'MANIPULATING'
  | 'COMPLETED'
export type SafetyLevel = 'NORMAL' | 'WARNING' | 'CRITICAL'

export interface BasEvent {
  id: number
  ts: number
  time: string
  type: string
  severity: Severity
  message: string
  activity: string
  state: string
  subject: string
  stability: number | null
  data: Record<string, unknown>
  acknowledged?: boolean
}

export interface Alert {
  id: number
  ts: number
  time: string
  type: string
  severity: Severity
  message: string
  subject: string
}

export interface TrackedObject {
  track_id: number
  label: string
  cls: string
  conf: number
  visible: boolean
  box: [number, number, number, number]
  state: string
  stability: number | null
  proximity: 'FAR' | 'NEAR' | 'CONTACT'
  restricted: boolean
  is_human: boolean
}

export type ZoneType = 'restricted' | 'workstation'

export interface Zone {
  id: string
  name: string
  type: ZoneType
  enabled: boolean
  points: [number, number][]
  active?: boolean
  level?: SafetyLevel
  hands?: string[]
}

export interface WorkflowStep {
  id: string
  name: string
  object: string
  action: 'interact' | 'move'
  status: 'pending' | 'active' | 'done'
  completed_at: number | null
}

export type ComponentState =
  | 'connected'
  | 'connecting'
  | 'stalled'
  | 'ended'
  | 'error'
  | 'ready'
  | 'active'
  | 'idle'
  | 'disabled'
  | 'muted'
  | 'starting'

export interface Health {
  camera: { state: ComponentState; detail: string }
  yolo: { state: ComponentState; detail: string }
  mediapipe: { state: ComponentState; detail: string }
  activity: { state: ComponentState; detail: string }
  safety: { state: ComponentState; detail: string }
  voice: { state: ComponentState; detail: string }
}

export interface CameraInfo {
  index: number
  available: boolean
  in_use: boolean
  detail: string
}

export interface Status {
  running: boolean
  paused: boolean
  mode: 'webcam' | 'file'
  source: string
  camera_ok: boolean
  camera_error: string
  is_file: boolean
  fps: number
  latency_ms: { detect: number; hands: number; logic: number; total: number }
  frame_size: [number, number]
  frames_processed: number
  last_frame_age_s: number | null
  processing_errors: number
  last_error: string
  max_processing_fps: number
  model: string
  detector_mode: string
  model_source?: string
  hand_backend: string
  voice_enabled: boolean
  voice_available: boolean
  uptime_s: number
  session_start: number
  log_file: string
  health: Health
  cameras: CameraInfo[]
  config_warnings: string[]
}

export interface SceneInfo {
  state: SceneState | 'WARNING' | 'CRITICAL'
  activity_state: SceneState
  safety_level: SafetyLevel
  label: string
  object: string | null
  hand: string | null
  stability: number | null
  detection_confidence: number | null
  since: number | null
  hand_tracking_lost: boolean
  objects_temporarily_lost: string[]
}

export interface LiveState {
  ts: number
  activity: SceneInfo
  objects: TrackedObject[]
  hands: { name: string; score: number; center: [number, number] }[]
  interactions: { hand: string; object: string; track_id: number; level: string; distance: number }[]
  zones: Zone[]
  human_present: boolean
  status: Status
  workflow: {
    name: string
    note: string
    current_index: number
    completed: boolean
    deviations: number
    steps: WorkflowStep[]
  }
  alerts: Alert[]
  counts: Record<Severity, number>
  app: { title?: string; subtitle?: string }
}
