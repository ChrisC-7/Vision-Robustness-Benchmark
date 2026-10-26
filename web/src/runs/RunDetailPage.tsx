import { Link, useParams } from 'react-router'
import { isNotFound, useRun, type RunDetail } from '../api/runs'
import { QueryError } from '../components/QueryStates'
import { MODEL_LABELS, seedLabel } from '../results/evaluations'
import { formatAccuracy } from '../results/format'

export function RunDetailPage() {
  // The route is /runs/*, so the whole run_dir, slashes included, is the
  // splat. It is never empty: the exact /runs route claims /runs and /runs/.
  const runDir = useParams()['*'] ?? ''
  const run = useRun(runDir)

  return (
    <>
      <Link to="/runs" className="back-link">
        ← All runs
      </Link>
      {run.isPending ? (
        <p role="status">Loading run…</p>
      ) : run.isError ? (
        isNotFound(run.error) ? (
          <div className="status" role="status">
            <p>
              No run is recorded for <code>{runDir}</code>.
            </p>
          </div>
        ) : (
          <QueryError onRetry={() => void run.refetch()} retrying={run.isFetching}>
            Couldn't load this run: {run.error.message}
          </QueryError>
        )
      ) : (
        <RunSummary run={run.data} />
      )}
    </>
  )
}

function RunSummary({ run }: { run: RunDetail }) {
  // train_config is whatever config.json held, typed as unknown values, so
  // each field has to be checked before it is used.
  const epochs = typeof run.train_config.epochs === 'number' ? run.train_config.epochs : undefined

  return (
    <article className="run-detail">
      <h3>{run.run_name}</h3>
      <p className="run-dir">
        <code>{run.run_dir}</code>
      </p>

      <dl className="stat-tiles">
        <div className="stat-tile">
          <dt>Best validation accuracy</dt>
          <dd>{run.best_val_accuracy === null ? '—' : `${formatAccuracy(run.best_val_accuracy)}%`}</dd>
        </div>
        <div className="stat-tile">
          <dt>Best epoch</dt>
          <dd>
            {run.best_epoch ?? '—'}
            {epochs !== undefined && <span className="stat-detail"> of {epochs}</span>}
          </dd>
        </div>
        <div className="stat-tile">
          <dt>Model</dt>
          <dd>{MODEL_LABELS[run.model_name]}</dd>
        </div>
        <div className="stat-tile">
          <dt>Training seed</dt>
          <dd>{seedLabel(run.training_seed)}</dd>
        </div>
      </dl>

      <h4>Training config</h4>
      <dl className="config-list">
        {Object.entries(run.train_config).map(([key, value]) => (
          <div key={key}>
            <dt>
              <code>{key}</code>
            </dt>
            <dd>{formatConfigValue(value)}</dd>
          </div>
        ))}
      </dl>
    </article>
  )
}

function formatConfigValue(value: unknown): string {
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean' || value === null) return String(value)
  return JSON.stringify(value)
}
