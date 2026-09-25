import { useEffect, useState } from 'react'
import type { LiveState, SceneState } from '../types'

const FLOW: { id: SceneState; label: string; hint: string }[] = [
  { id: 'IDLE', label: 'Idle', hint: 'No person in view' },
  { id: 'PERSON_DETECTED', label: 'Person', hint: 'Person or hand detected for several frames' },
  { id: 'APPROACHING', label: 'Approach', hint: 'Hand inside the workstation zone' },
  { id: 'HAND_NEAR_EQUIPMENT', label: 'Hand near', hint: 'Hand near a stand-in object for several frames' },
  { id: 'INTERACTING', label: 'Interact', hint: 'Sustained 2D hand-object overlap (temporally confirmed)' },
  { id: 'MANIPULATING', label: 'Move', hint: 'Object displaced or held while in contact' },
  { id: 'COMPLETED', label: 'Complete', hint: 'Hand released the object for several frames' },
]

const STABILITY_INFO =
  'Derived from detection confidence and temporal consistency; not a trained activity-classifier probability. ' +
  'Activity Stability = share of the last frames whose hand-object evidence agrees with the current state.'

function useNow(ms = 250) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), ms)
    return () => window.clearInterval(t)
  }, [ms])
  return now
}

function Meter({ label, value, info }: { label: string; value: number | null; info: string }) {
  return (
    <div className="meter">
      <div className="meter__row">
        <span>
          {label}{' '}
          <span className="info" title={info} aria-label={info} tabIndex={0}>
            ⓘ
          </span>
        </span>
        <span className="num">{value !== null ? `${Math.round(value * 100)}%` : '-'}</span>
      </div>
      <div className="bar">
        <div className="bar__fill" style={{ width: `${(value ?? 0) * 100}%` }} />
      </div>
    </div>
  )
}

export function ActivityPanel({ state }: { state: LiveState | null }) {
  const now = useNow()
  const act = state?.activity
  const current = act?.activity_state ?? 'IDLE'
  const idx = FLOW.findIndex((f) => f.id === current)
  const level = act?.safety_level ?? 'NORMAL'
  const since = act?.since ? Math.max(0, now / 1000 - act.since) : null

  return (
    <section className="panel activity">
      <header className="panel__head">
        <h2>Current activity</h2>
        <span className={`badge state-${(act?.state ?? 'IDLE').toLowerCase()}`}>{act?.state ?? 'IDLE'}</span>
      </header>

      <div className="activity__label">{act?.label ?? 'Connecting…'}</div>

      {(act?.hand_tracking_lost || (act?.objects_temporarily_lost?.length ?? 0) > 0) && (
        <div className="notices">
          {act?.hand_tracking_lost && <span className="notice">HAND TRACKING LOST - holding state</span>}
          {act?.objects_temporarily_lost?.map((o) => (
            <span key={o} className="notice notice--soft">
              OBJECT TEMPORARILY LOST: {o}
            </span>
          ))}
        </div>
      )}

      <dl className="activity__meta">
        <div>
          <dt>Object (stand-in)</dt>
          <dd>{act?.object ?? '-'}</dd>
        </div>
        <div>
          <dt>Hand</dt>
          <dd>{act?.hand || '-'}</dd>
        </div>
        <div>
          <dt>In state</dt>
          <dd className="num">{since !== null ? `${since.toFixed(1)} s` : '-'}</dd>
        </div>
        <div>
          <dt>Person present</dt>
          <dd>{state ? (state.human_present ? 'Yes' : 'No') : '-'}</dd>
        </div>
      </dl>

      <Meter label="Activity Stability" value={act?.stability ?? null} info={STABILITY_INFO} />
      <Meter
        label="Object Detection Confidence"
        value={act?.detection_confidence ?? null}
        info="Confidence of the pretrained YOLO detector for the object involved (per-frame detection score, not an activity score)."
      />

      <h3 className="subhead">Temporal state machine</h3>
      <ol className="fsm" aria-label="Activity state machine">
        {FLOW.map((f, i) => (
          <li
            key={f.id}
            className={`fsm__node ${i === idx ? 'is-current' : ''} ${i < idx ? 'is-past' : ''} state-${f.id.toLowerCase()}`}
            title={f.hint}
          >
            <span className="fsm__dot" />
            <span className="fsm__label">{f.label}</span>
          </li>
        ))}
      </ol>
      <div className={`fsm__branch fsm__branch--${level.toLowerCase()}`}>
        <span className="fsm__dot" />
        Safety: <b>{level}</b>
        <span className="fsm__branch-note">
          {level === 'NORMAL'
            ? 'rule-based safety engine'
            : level === 'WARNING'
              ? 'warning condition active'
              : 'critical condition active'}
        </span>
      </div>
    </section>
  )
}
