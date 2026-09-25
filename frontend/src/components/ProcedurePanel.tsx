import { api } from '../api'
import type { LiveState } from '../types'

/** Representative Demo Workflow - expected-sequence validation (not an official BAS/ISRO protocol). */
export function WorkflowPanel({ state }: { state: LiveState | null }) {
  const w = state?.workflow
  const done = w?.steps.filter((s) => s.status === 'done').length ?? 0
  const total = w?.steps.length ?? 0
  return (
    <section className="panel procedure">
      <header className="panel__head">
        <h2>Representative Demo Workflow</h2>
        <button className="btn btn--ghost" onClick={() => api.resetWorkflow()}>
          Reset
        </button>
      </header>
      <p className="note">
        This workflow is a representative demonstration sequence created to validate the prototype's
        activity-recognition architecture. It is not an official BAS/ISRO protocol.
      </p>
      <div className="procedure__title">
        <span>Progress</span>
        <span className="num">
          {done}/{total}
        </span>
      </div>
      <div className="bar">
        <div className="bar__fill bar__fill--ok" style={{ width: total ? `${(done / total) * 100}%` : 0 }} />
      </div>
      <ol className="steps">
        {w?.steps.map((s) => (
          <li key={s.id} className={`step step--${s.status}`}>
            <span className="step__mark">{s.status === 'done' ? '✓' : s.id.replace(/\D/g, '')}</span>
            <div>
              <div className="step__name">{s.name}</div>
              <div className="step__meta">
                {s.action === 'move' ? 'Pick up & relocate' : 'Handle'} · {s.object}
                {s.completed_at && ` · ${new Date(s.completed_at * 1000).toLocaleTimeString()}`}
              </div>
            </div>
          </li>
        ))}
      </ol>
      <div className="procedure__foot">
        {w?.completed ? <span className="tag tag--ok">Demo workflow complete</span> : <span className="tag">In progress</span>}
        <span className={`tag ${w?.deviations ? 'tag--warn' : ''}`}>
          {w?.deviations ?? 0} sequence deviation{w?.deviations === 1 ? '' : 's'}
        </span>
      </div>
    </section>
  )
}
