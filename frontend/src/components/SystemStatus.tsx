import { useState } from 'react'
import { api } from '../api'
import type { ComponentState, LiveState } from '../types'
import type { Sample } from '../useLiveState'

function fmtDur(s: number) {
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

/** Green only with evidence that the component works; amber = waiting/muted; red = error. */
function tone(s: ComponentState | undefined): 'ok' | 'wait' | 'bad' {
  if (s === 'connected' || s === 'active' || s === 'ready') return 'ok'
  if (s === 'error' || s === 'stalled' || s === 'ended' || s === undefined) return 'bad'
  return 'wait'
}

function Sparkline({ values, max }: { values: number[]; max: number }) {
  if (values.length < 2) return <svg className="spark" viewBox="0 0 100 28" />
  const pts = values
    .map((v, i) => `${(i / (values.length - 1)) * 100},${28 - Math.min(v / max, 1) * 26 - 1}`)
    .join(' ')
  return (
    <svg className="spark" viewBox="0 0 100 28" preserveAspectRatio="none" aria-hidden>
      <polyline points={`0,28 ${pts} 100,28`} className="spark__area" />
      <polyline points={pts} className="spark__line" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

const COMPONENTS: { key: keyof NonNullable<LiveState['status']['health']>; name: string }[] = [
  { key: 'camera', name: 'Camera' },
  { key: 'yolo', name: 'YOLO' },
  { key: 'mediapipe', name: 'MediaPipe' },
  { key: 'activity', name: 'Activity engine' },
  { key: 'safety', name: 'Safety engine' },
  { key: 'voice', name: 'Voice alert' },
]

export function SystemStatus({
  state,
  connected,
  samples,
  browserVoice,
}: {
  state: LiveState | null
  connected: boolean
  samples: Sample[]
  browserVoice: boolean
}) {
  const s = state?.status
  const h = s?.health
  const lat = s?.latency_ms
  const [voiceMsg, setVoiceMsg] = useState('')
  const fpsVals = samples.map((x) => x.fps)
  const allOk = connected && !!h && COMPONENTS.every((c) => tone(h[c.key].state) === 'ok')

  return (
    <section className="panel system">
      <header className="panel__head">
        <h2>System health</h2>
        <span className={`pill ${allOk ? 'pill--ok' : 'pill--bad'}`}>
          <span className={`dot ${allOk ? 'dot--ok' : 'dot--bad'}`} />
          {!connected ? 'Backend offline' : allOk ? 'All components ready' : 'Check components'}
        </span>
      </header>

      <ul className="health">
        {COMPONENTS.map((c) => {
          const item = h?.[c.key]
          const t = connected ? tone(item?.state) : 'bad'
          return (
            <li key={c.key} className={`health__item health--${t}`}>
              <span className="health__name">
                <span className={`dot dot--${t === 'ok' ? 'ok' : t === 'bad' ? 'bad' : 'wait'}`} /> {c.name}
              </span>
              <span className="health__state">{connected ? (item?.state ?? '-').toUpperCase() : 'OFFLINE'}</span>
              <span className="health__detail" title={item?.detail}>
                {c.key === 'voice' && browserVoice ? 'backend TTS unavailable - using browser voice fallback' : item?.detail}
              </span>
            </li>
          )
        })}
        <li className="health__item health--ok">
          <span className="health__name">FPS</span>
          <span className="health__state num big">{s ? s.fps.toFixed(1) : '--'}</span>
          <span className="health__detail">cap {s?.max_processing_fps ?? '-'} FPS</span>
        </li>
      </ul>

      <div className="metrics">
        <div className="metric">
          <div className="metric__label">Throughput</div>
          <div className="metric__value num">
            {s ? s.fps.toFixed(1) : '--'} <small>FPS</small>
          </div>
          <Sparkline values={fpsVals} max={Math.max(15, ...fpsVals)} />
        </div>
        <div className="metric">
          <div className="metric__label">Inference (avg per run)</div>
          <div className="metric__value metric__value--sm num">
            YOLO {lat ? lat.detect.toFixed(0) : '--'} ms · Hands {lat ? lat.hands.toFixed(0) : '--'} ms · Logic{' '}
            {lat ? lat.logic.toFixed(1) : '--'} ms
          </div>
          <Sparkline values={samples.map((x) => x.latency)} max={Math.max(200, ...samples.map((x) => x.latency))} />
        </div>
        <div className="metric">
          <div className="metric__label">Frame processing</div>
          <div className="metric__value metric__value--sm num">
            {s?.frames_processed.toLocaleString() ?? '-'} frames · last{' '}
            {s?.last_frame_age_s !== null && s?.last_frame_age_s !== undefined ? `${s.last_frame_age_s.toFixed(1)}s ago` : '-'}
          </div>
          <div className={`small ${s?.processing_errors ? 'text-bad' : 'muted'}`}>
            {s?.processing_errors ? `${s.processing_errors} errors - last: ${s.last_error}` : 'no processing errors'}
          </div>
        </div>
      </div>

      <dl className="kv">
        <div><dt>Mode</dt><dd>{s ? (s.mode === 'webcam' ? `LIVE webcam ${s.source}` : `file ${s.source}`) : '-'}</dd></div>
        <div><dt>Detector</dt><dd>{s ? `${s.model} (${s.model_source || (s.detector_mode === 'custom' ? 'CUSTOM MODEL' : 'BASE PRETRAINED MODEL')})` : '-'}</dd></div>
        <div><dt>Frame</dt><dd className="num">{s ? `${s.frame_size[0]}×${s.frame_size[1]}` : '-'}</dd></div>
        <div><dt>Uptime</dt><dd className="num">{s ? fmtDur(s.uptime_s) : '-'}</dd></div>
        <div><dt>Session log (metadata only)</dt><dd className="small">{s?.log_file ?? '-'}</dd></div>
        <div>
          <dt>Webcams found</dt>
          <dd className="small">
            {s?.cameras?.length
              ? s.cameras.filter((c) => c.available).map((c) => `#${c.index}${c.in_use ? ' (active)' : ''}`).join(', ') || 'none'
              : 'scanning…'}
          </dd>
        </div>
      </dl>
      {!!s?.config_warnings?.length && <div className="form-error">Config: {s.config_warnings.join('; ')}</div>}

      <div className="toolbar">
        <button className="btn" onClick={() => api.voice(!s?.voice_enabled)} disabled={!s}>
          {s?.voice_enabled ? '🔊 Voice on' : '🔇 Voice muted'}
        </button>
        <button
          className="btn btn--ghost"
          disabled={!s}
          onClick={async () => {
            const r = await api.voiceTest()
            setVoiceMsg(r.available ? 'Test phrase queued (listen for "Voice alert system ready")' : `Backend TTS unavailable: ${r.error}`)
          }}
        >
          Test voice
        </button>
        {voiceMsg && <span className="small muted">{voiceMsg}</span>}
      </div>
    </section>
  )
}
