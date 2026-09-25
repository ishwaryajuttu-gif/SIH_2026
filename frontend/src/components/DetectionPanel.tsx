import type { LiveState } from '../types'

export function DetectionPanel({ state }: { state: LiveState | null }) {
  const objs = [...(state?.objects ?? [])].sort(
    (a, b) => Number(a.is_human) - Number(b.is_human) || a.track_id - b.track_id,
  )
  const hands = state?.hands ?? []
  return (
    <section className="panel detections">
      <header className="panel__head">
        <h2>Detections</h2>
        <span className="muted num">
          {objs.length} obj · {hands.length} hand{hands.length === 1 ? '' : 's'}
        </span>
      </header>
      <p className="note">
        {state?.status?.model_source?.includes('CUSTOM MODEL')
          ? 'Custom YOLO detector active: recognizing BAS experimental stand-in objects (Sample Container, Culture Vessel, Data Tablet, Restricted Tool).'
          : state?.status?.model_source?.includes('fallback')
            ? 'FALLBACK: Base pretrained YOLO (COCO) active because custom model weights were not found. Detecting equipment stand-ins.'
            : 'Pretrained YOLO (COCO) detecting representative demonstration objects - BAS equipment stand-ins, not real BAS hardware.'}
      </p>
      <table className="table">
        <thead>
          <tr>
            <th>ID</th>
            <th>Object (stand-in)</th>
            <th title="Per-frame YOLO detection score">Detection conf.</th>
            <th>Hand (2D)</th>
            <th>State</th>
          </tr>
        </thead>
        <tbody>
          {objs.length === 0 && (
            <tr>
              <td colSpan={5} className="empty">
                No objects detected
              </td>
            </tr>
          )}
          {objs.map((o) => (
            <tr key={o.track_id} className={o.visible ? '' : 'is-hidden'}>
              <td className="num">#{o.track_id}</td>
              <td>
                <div className="obj">
                  {o.label}
                  {o.restricted && <span className="tag tag--crit">restricted</span>}
                </div>
                <div className="muted small">
                  YOLO class: {o.cls}
                  {!o.visible && ' · temporarily lost'}
                </div>
              </td>
              <td>
                <div className="mini-bar">
                  <div style={{ width: `${o.conf * 100}%` }} />
                </div>
                <span className="num small">{Math.round(o.conf * 100)}%</span>
              </td>
              <td>
                <span className={`prox prox--${o.proximity.toLowerCase()}`}>{o.is_human ? '-' : o.proximity}</span>
              </td>
              <td>
                <span className={`badge badge--sm state-${o.state.toLowerCase()}`}>
                  {o.state === 'HAND_NEAR_EQUIPMENT' ? 'HAND NEAR' : o.state}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {hands.length > 0 && (
        <div className="hands">
          {hands.map((h, i) => (
            <span key={i} className="chip">
              ✋ {h.name} <span className="num">{Math.round(h.score * 100)}%</span>
            </span>
          ))}
        </div>
      )}
    </section>
  )
}
