import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { ActivityPanel } from './components/ActivityPanel'
import { AlertPanel, CriticalBanner } from './components/AlertPanel'
import { DetectionPanel } from './components/DetectionPanel'
import { EventTimeline } from './components/EventTimeline'
import { LiveFeed } from './components/LiveFeed'
import { WorkflowPanel } from './components/ProcedurePanel'
import { SystemStatus } from './components/SystemStatus'
import type { BasEvent, LiveState } from './types'
import { useLiveState } from './useLiveState'

function Clock({ sessionStart }: { sessionStart?: number }) {
  const [now, setNow] = useState(new Date())
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 1000)
    return () => window.clearInterval(t)
  }, [])
  const met = sessionStart ? Math.max(0, Math.floor(now.getTime() / 1000 - sessionStart)) : 0
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    <div className="clocks">
      <div>
        <span className="clocks__k">Local</span>
        <span className="num">{now.toLocaleTimeString([], { hour12: false })}</span>
      </div>
      <div>
        <span className="clocks__k">Session</span>
        <span className="num">
          {pad(Math.floor(met / 3600))}:{pad(Math.floor((met % 3600) / 60))}:{pad(met % 60)}
        </span>
      </div>
    </div>
  )
}

// Same mapping as backend/app/voice.py - used only if the backend's offline TTS failed to start.
const PHRASE_KEY: Record<string, string> = {
  RESTRICTED_ZONE_ENTRY: 'restricted_zone',
  RESTRICTED_ZONE_CRITICAL: 'critical',
  RESTRICTED_OBJECT: 'critical',
  UNEXPECTED_MOVEMENT: 'unexpected',
  OUT_OF_SEQUENCE: 'unexpected',
  OBJECT_MISSING: 'unexpected',
  CAMERA_SHIFT: 'unexpected',
  PROLONGED_INTERACTION: 'unexpected',
}

/** Browser speech fallback (local OS voices only) when backend TTS reports an error. */
function useBrowserVoiceFallback(state: LiveState | null, events: BasEvent[]) {
  const active = !!state && state.status.voice_enabled && state.status.health?.voice.state === 'error'
  const phrases = useRef<Record<string, string>>({})
  const lastId = useRef(0)
  const lastSpoken = useRef<Record<string, number>>({})

  useEffect(() => {
    api.config().then((c) => (phrases.current = c.voice?.phrases ?? {})).catch(() => {})
  }, [])

  useEffect(() => {
    if (!events.length) return
    const fresh = events.filter((e) => e.id > lastId.current)
    lastId.current = events[events.length - 1].id
    if (!active || !('speechSynthesis' in window)) return
    for (const e of fresh) {
      const text = phrases.current[PHRASE_KEY[e.type]]
      if (!text || Date.now() - (lastSpoken.current[text] ?? 0) < 8000) continue
      lastSpoken.current[text] = Date.now()
      const u = new SpeechSynthesisUtterance(text)
      const local = window.speechSynthesis.getVoices().find((v) => v.localService && v.lang.startsWith('en'))
      if (local) u.voice = local
      window.speechSynthesis.speak(u)
    }
  }, [events, active])
  return active
}

export default function App() {
  const { state, events, connected, samples } = useLiveState()
  const counts = state?.counts
  const browserVoice = useBrowserVoiceFallback(state, events)

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 32 32" className="brand__logo" aria-hidden>
            <circle cx="16" cy="16" r="9" fill="none" stroke="currentColor" strokeWidth="2.5" />
            <circle cx="16" cy="16" r="3.5" className="brand__core" />
            <path d="M16 3v5M16 24v5M3 16h5M24 16h5" stroke="currentColor" strokeWidth="2" />
          </svg>
          <div>
            <h1>
              {state?.app?.title ?? 'BAS-HAR Prototype'}{' '}
              <span className="proto" title="Pretrained perception models + rule-based temporal reasoning, demonstrated with stand-in objects. Not trained or validated on BAS mission data.">
                Prototype / Controlled Demonstration
              </span>
            </h1>
            <p>{state?.app?.subtitle ?? 'AI-assisted Human Activity Recognition'}</p>
          </div>
        </div>
        <div className="topbar__right">
          <div className="counters" aria-label="Event counts">
            <span className="counter sev-critical" title="Critical events">
              {counts?.critical ?? 0}
            </span>
            <span className="counter sev-warning" title="Warnings">
              {counts?.warning ?? 0}
            </span>
            <span className="counter sev-success" title="Completed activities / workflow steps">
              {counts?.success ?? 0}
            </span>
          </div>
          <Clock sessionStart={state?.status?.session_start} />
          <span className={`pill ${connected ? 'pill--ok' : 'pill--bad'}`}>
            <span className={`dot ${connected ? 'dot--ok' : 'dot--bad'}`} />
            {connected ? 'Backend connected' : 'Backend offline'}
          </span>
        </div>
      </header>

      <CriticalBanner state={state} />

      <main className="grid">
        <div className="col-main">
          <LiveFeed state={state} connected={connected} />
          <div className="row-2">
            <DetectionPanel state={state} />
            <WorkflowPanel state={state} />
          </div>
        </div>
        <div className="col-side">
          <ActivityPanel state={state} />
          <AlertPanel state={state} />
          <EventTimeline events={events} />
        </div>
        <div className="col-full">
          <SystemStatus state={state} connected={connected} samples={samples} browserVoice={browserVoice} />
        </div>
      </main>
      <footer className="foot">
        The current prototype uses pretrained computer-vision models for perception and a controlled demonstration
        environment for activity recognition. The architecture is designed so that mission-specific BAS datasets can
        later be used for domain-specific fine-tuning and validation. All processing runs locally; no video is recorded.
      </footer>
    </div>
  )
}
