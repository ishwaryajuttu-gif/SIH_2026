import { useEffect, useRef, useState } from 'react'
import type { BasEvent, LiveState } from './types'

const MAX_EVENTS = 400
const MAX_SAMPLES = 90

export interface Sample {
  t: number
  fps: number
  latency: number
}

/** Subscribes to the backend WebSocket, reconnecting automatically. */
export function useLiveState() {
  const [state, setState] = useState<LiveState | null>(null)
  const [events, setEvents] = useState<BasEvent[]>([])
  const [connected, setConnected] = useState(false)
  const [samples, setSamples] = useState<Sample[]>([])
  const lastSample = useRef(0)

  useEffect(() => {
    let ws: WebSocket | null = null
    let retry: number | undefined
    let closed = false

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws/state`)
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closed) retry = window.setTimeout(connect, 1500)
      }
      ws.onerror = () => ws?.close()
      ws.onmessage = (msg) => {
        const data = JSON.parse(msg.data)
        if (data.kind === 'history') {
          setEvents(data.events)
          return
        }
        const s: LiveState = data.state
        setState(s)
        if (data.events?.length) {
          setEvents((prev) => {
            const seen = new Set(prev.map((e) => e.id))
            const fresh = (data.events as BasEvent[]).filter((e) => !seen.has(e.id))
            return [...prev, ...fresh].slice(-MAX_EVENTS)
          })
        }
        const now = Date.now()
        if (s.status && now - lastSample.current > 500) {
          lastSample.current = now
          setSamples((prev) =>
            [...prev, { t: now, fps: s.status.fps, latency: s.status.latency_ms.total }].slice(-MAX_SAMPLES),
          )
        }
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(retry)
      ws?.close()
    }
  }, [])

  return { state, events, connected, samples }
}
