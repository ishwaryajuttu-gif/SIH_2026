import { api } from '../api'
import type { LiveState } from '../types'

const ICON: Record<string, string> = { critical: '⛔', warning: '⚠', info: 'ℹ', success: '✓' }

export function CriticalBanner({ state }: { state: LiveState | null }) {
  const crit = (state?.alerts ?? []).filter((a) => a.severity === 'critical')
  if (!crit.length) return null
  const a = crit[crit.length - 1]
  return (
    <div className="banner" role="alert">
      <span className="banner__icon">⛔</span>
      <span className="banner__text">
        <b>{a.time}</b> {a.message}
      </span>
      <button className="btn btn--light" onClick={() => api.ackAll()}>
        Acknowledge
      </button>
    </div>
  )
}

export function AlertPanel({ state }: { state: LiveState | null }) {
  const alerts = [...(state?.alerts ?? [])].reverse()
  return (
    <section className="panel alerts">
      <header className="panel__head">
        <h2>
          Alerts <span className="count">{alerts.length}</span>
        </h2>
        {alerts.length > 0 && (
          <button className="btn btn--ghost" onClick={() => api.ackAll()}>
            Ack all
          </button>
        )}
      </header>
      {alerts.length === 0 ? (
        <div className="empty">
          <span className="dot dot--ok" /> No active alerts
        </div>
      ) : (
        <ul className="alert-list">
          {alerts.map((a) => (
            <li key={a.id} className={`alert alert--${a.severity}`}>
              <span className="alert__icon" aria-hidden>
                {ICON[a.severity]}
              </span>
              <div className="alert__body">
                <div className="alert__msg">{a.message}</div>
                <div className="alert__meta">
                  <span className="num">{a.time}</span> · {a.type.replace(/_/g, ' ').toLowerCase()}
                </div>
              </div>
              <button className="btn btn--ghost btn--sm" onClick={() => api.ack(a.id)} title="Acknowledge">
                Ack
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
