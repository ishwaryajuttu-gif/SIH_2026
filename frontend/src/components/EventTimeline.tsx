import { useMemo, useState } from 'react'
import type { BasEvent, Severity } from '../types'

const FILTERS: { id: 'all' | Severity; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'critical', label: 'Critical' },
  { id: 'warning', label: 'Warning' },
  { id: 'success', label: 'Activity' },
  { id: 'info', label: 'Info' },
]

export function EventTimeline({ events }: { events: BasEvent[] }) {
  const [filter, setFilter] = useState<'all' | Severity>('all')
  const list = useMemo(
    () => [...events].reverse().filter((e) => filter === 'all' || e.severity === filter),
    [events, filter],
  )
  return (
    <section className="panel timeline">
      <header className="panel__head">
        <h2>Event timeline</h2>
        <div className="toolbar">
          <a className="btn btn--ghost" href="/api/events/export?format=csv" download>
            ⤓ CSV
          </a>
          <a className="btn btn--ghost" href="/api/events/export?format=json" download>
            ⤓ JSON
          </a>
        </div>
      </header>
      <div className="filters" role="tablist">
        {FILTERS.map((f) => (
          <button
            key={f.id}
            role="tab"
            aria-selected={filter === f.id}
            className={`filter ${filter === f.id ? 'is-on' : ''} sev-${f.id}`}
            onClick={() => setFilter(f.id)}
          >
            {f.label}
          </button>
        ))}
      </div>
      <ol className="events">
        {list.length === 0 && <li className="empty">No events yet</li>}
        {list.map((e) => (
          <li key={e.id} className={`event sev-${e.severity}`}>
            <span className="event__time num">{e.time}</span>
            <span className="event__rail" />
            <div className="event__body">
              <div className="event__msg">{e.message}</div>
              <div className="event__type">
                {e.type}
                {e.state && ` · ${e.state}`}
                {e.subject && ` · ${e.subject}`}
                {e.stability !== null && e.stability !== undefined && ` · stability ${Math.round(e.stability * 100)}%`}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
