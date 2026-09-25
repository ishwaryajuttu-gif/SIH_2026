import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { LiveState, Zone, ZoneType } from '../types'

interface Props {
  state: LiveState | null
  connected: boolean
}

type Shape = 'rect' | 'poly'
type Pt = [number, number]

const r4 = (v: number) => +Math.min(1, Math.max(0, v)).toFixed(4)

export function LiveFeed({ state, connected }: Props) {
  const [editing, setEditing] = useState(false)
  const [draftZones, setDraftZones] = useState<Zone[]>([])
  const [draftPts, setDraftPts] = useState<Pt[]>([])
  const [dragStart, setDragStart] = useState<Pt | null>(null)
  const [dragEnd, setDragEnd] = useState<Pt | null>(null)
  const [shape, setShape] = useState<Shape>('rect')
  const [zoneType, setZoneType] = useState<ZoneType>('restricted')
  const [videos, setVideos] = useState<string[]>([])
  const [imgKey, setImgKey] = useState(0)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const svgRef = useRef<SVGSVGElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const status = state?.status
  const [fw, fh] = status?.frame_size?.[0] ? status.frame_size : [16, 9]
  const cameras = status?.cameras ?? []
  const current = status ? `${status.mode}:${status.source}` : ''

  useEffect(() => {
    api.videos().then(setVideos).catch(() => {})
  }, [status?.source])

  const run = async (label: string, fn: () => Promise<unknown>) => {
    setBusy(label)
    setError('')
    try {
      await fn()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy('')
    }
  }

  const toNorm = (e: React.PointerEvent | React.MouseEvent): Pt | null => {
    if (!svgRef.current) return null
    const r = svgRef.current.getBoundingClientRect()
    return [r4((e.clientX - r.left) / r.width), r4((e.clientY - r.top) / r.height)]
  }

  const addZone = (points: Pt[]) => {
    const n = draftZones.filter((z) => z.type === zoneType).length + 1
    const name = zoneType === 'restricted' ? (n === 1 ? 'RESTRICTED ZONE' : `RESTRICTED ZONE ${n}`) : 'Workstation (tray)'
    setDraftZones((z) => [...z, { id: `zone-${Date.now()}`, name, type: zoneType, enabled: true, points }])
  }

  const onDown = (e: React.PointerEvent) => {
    if (!editing || shape !== 'rect') return
    const p = toNorm(e)
    setDragStart(p)
    setDragEnd(p)
  }
  const onMove = (e: React.PointerEvent) => {
    if (dragStart) setDragEnd(toNorm(e))
  }
  const onUp = () => {
    if (dragStart && dragEnd) {
      const [x1, y1] = [Math.min(dragStart[0], dragEnd[0]), Math.min(dragStart[1], dragEnd[1])]
      const [x2, y2] = [Math.max(dragStart[0], dragEnd[0]), Math.max(dragStart[1], dragEnd[1])]
      if (x2 - x1 > 0.02 && y2 - y1 > 0.02) addZone([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    }
    setDragStart(null)
    setDragEnd(null)
  }
  const onClick = (e: React.MouseEvent) => {
    if (!editing || shape !== 'poly') return
    const p = toNorm(e)
    if (p) setDraftPts((pts) => [...pts, p])
  }

  const startEdit = () => {
    setDraftZones((state?.zones ?? []).map(({ id, name, type, enabled, points }) => ({ id, name, type, enabled, points })))
    setDraftPts([])
    setEditing(true)
  }

  const changeSource = (v: string) => {
    if (v === '__upload') {
      fileRef.current?.click()
      return
    }
    const [kind, ...rest] = v.split(':')
    run('Switching source…', async () => {
      await api.setSource(kind as 'webcam' | 'file', rest.join(':'))
      setImgKey((k) => k + 1)
    })
  }

  const onUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    run(`Uploading ${f.name}…`, async () => {
      await api.upload(f)
      setVideos(await api.videos())
      setImgKey((k) => k + 1)
    }).finally(() => (e.target.value = ''))
  }

  const poly = (pts: Pt[]) => pts.map((p) => p.join(',')).join(' ')
  const act = state?.activity
  const alarm = act?.safety_level === 'CRITICAL' ? 'feed--alarm' : act?.safety_level === 'WARNING' ? 'feed--warn' : ''
  const camState = status?.health?.camera

  return (
    <section className={`panel feed ${alarm}`}>
      <header className="panel__head">
        <h2>
          Live monitoring
          <span className={`mode ${status?.mode === 'file' ? 'mode--file' : ''}`}>
            {status?.mode === 'file' ? 'VIDEO FILE (fallback)' : 'LIVE WEBCAM'}
          </span>
        </h2>
        <div className="toolbar">
          <select value={current} onChange={(e) => changeSource(e.target.value)} aria-label="Video source">
            <optgroup label="Live webcam (primary)">
              {(cameras.length ? cameras : [{ index: 0, available: true, in_use: false, detail: 'scanning…' }]).map((c) => (
                <option key={c.index} value={`webcam:${c.index}`} disabled={!c.available && !c.in_use}>
                  Webcam {c.index}
                  {c.in_use ? ' (active)' : c.available ? '' : ' (not found)'}
                </option>
              ))}
              {status?.mode === 'webcam' && !cameras.some((c) => String(c.index) === status.source) && (
                <option value={current}>Webcam {status.source}</option>
              )}
            </optgroup>
            <optgroup label="Video file (fallback)">
              {videos.map((v) => (
                <option key={v} value={`file:${v}`}>
                  {v.split('/').pop()}
                </option>
              ))}
              <option value="__upload">Upload video…</option>
            </optgroup>
          </select>
          <button className="btn btn--ghost" onClick={() => run('Scanning cameras…', api.rescanCameras)} title="Detect webcams">
            ⟳ Cameras
          </button>
          <input ref={fileRef} type="file" accept="video/*" hidden onChange={onUpload} />
          <button className="btn" onClick={() => api.control(status?.paused ? 'resume' : 'pause')} disabled={!connected}>
            {status?.paused ? '▶ Resume' : '❚❚ Pause'}
          </button>
          {!editing && (
            <button className="btn" onClick={startEdit} disabled={!connected}>
              ⬚ Zones
            </button>
          )}
        </div>
      </header>

      {editing && (
        <div className="zone-editor">
          <div className="toolbar">
            <label className="field">
              Type
              <select value={zoneType} onChange={(e) => setZoneType(e.target.value as ZoneType)}>
                <option value="restricted">Restricted zone</option>
                <option value="workstation">Workstation (tray)</option>
              </select>
            </label>
            <label className="field">
              Shape
              <select value={shape} onChange={(e) => { setShape(e.target.value as Shape); setDraftPts([]) }}>
                <option value="rect">Rectangle (drag)</option>
                <option value="poly">Polygon (click points)</option>
              </select>
            </label>
            {shape === 'poly' && (
              <button className="btn" disabled={draftPts.length < 3} onClick={() => { addZone(draftPts); setDraftPts([]) }}>
                Close shape
              </button>
            )}
            <span className="spacer" />
            <button
              className="btn btn--primary"
              onClick={() => run('Saving zones…', async () => { await api.saveZones(draftZones); setEditing(false) })}
            >
              Save zones
            </button>
            <button className="btn btn--ghost" onClick={() => setEditing(false)}>
              Cancel
            </button>
          </div>
          <ul className="zone-list">
            {draftZones.length === 0 && <li className="muted small">No zones - draw one on the video.</li>}
            {draftZones.map((z, i) => (
              <li key={z.id} className={`zone-item zone-item--${z.type}`}>
                <input
                  className="zone-name"
                  value={z.name}
                  aria-label="Zone name"
                  onChange={(e) => setDraftZones((all) => all.map((x, j) => (j === i ? { ...x, name: e.target.value } : x)))}
                />
                <span className="tag">{z.type}</span>
                <label className="small">
                  <input
                    type="checkbox"
                    checked={z.enabled}
                    onChange={(e) => setDraftZones((all) => all.map((x, j) => (j === i ? { ...x, enabled: e.target.checked } : x)))}
                  />{' '}
                  active
                </label>
                <button className="btn btn--ghost btn--sm" onClick={() => setDraftZones((all) => all.filter((_, j) => j !== i))}>
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="feed__stage" style={{ aspectRatio: `${fw} / ${fh}` }}>
        {connected && status?.frames_processed ? (
          <img key={`${imgKey}-${connected}`} src={`/video_feed?k=${imgKey}`} alt="Annotated live camera feed" />
        ) : (
          <div className="feed__offline">
            <span className="dot dot--bad" />
            {!connected
              ? 'Waiting for the backend on http://localhost:8000 …'
              : camState?.state === 'error'
                ? camState.detail
                : camState?.state === 'connecting'
                  ? 'Connecting to camera…'
                  : 'Waiting for the first processed frame…'}
          </div>
        )}

        <svg
          ref={svgRef}
          className={`feed__svg ${editing ? 'is-editing' : ''}`}
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={onUp}
          onClick={onClick}
        >
          {editing &&
            draftZones.map((z) => (
              <polygon
                key={z.id}
                points={poly(z.points)}
                className={`zone-draft zone-draft--${z.type} ${z.enabled ? '' : 'is-off'}`}
                vectorEffect="non-scaling-stroke"
              />
            ))}
          {editing && dragStart && dragEnd && (
            <rect
              x={Math.min(dragStart[0], dragEnd[0])}
              y={Math.min(dragStart[1], dragEnd[1])}
              width={Math.abs(dragEnd[0] - dragStart[0])}
              height={Math.abs(dragEnd[1] - dragStart[1])}
              className="zone-open"
              vectorEffect="non-scaling-stroke"
            />
          )}
          {editing && draftPts.length > 0 && (
            <>
              <polyline points={poly(draftPts)} className="zone-open" vectorEffect="non-scaling-stroke" />
              {draftPts.map((p, i) => (
                <circle key={i} cx={p[0]} cy={p[1]} r={0.007} className="zone-pt" />
              ))}
            </>
          )}
        </svg>

        <div className="feed__hud feed__hud--tl">
          <span className={`live ${status?.paused ? 'live--paused' : ''} ${status?.mode === 'file' ? 'live--file' : ''}`}>
            {status?.paused ? 'PAUSED' : status?.mode === 'file' ? 'FILE' : 'LIVE'}
          </span>
          <span className="chip num">{status ? `${status.fps.toFixed(1)} FPS` : '-- FPS'}</span>
        </div>
        {act && (
          <div className={`feed__hud feed__hud--bl sev-${act.safety_level.toLowerCase()}`}>
            <span className="feed__act">{act.label}</span>
          </div>
        )}
        {editing && (
          <div className="feed__hint">
            {shape === 'rect' ? 'Drag on the video to draw a rectangle.' : 'Click points on the video, then Close shape.'} Then
            press <b>Save zones</b>.
          </div>
        )}
        {busy && <div className="feed__busy">{busy}</div>}
      </div>
      {error && <div className="form-error">{error}</div>}
    </section>
  )
}
