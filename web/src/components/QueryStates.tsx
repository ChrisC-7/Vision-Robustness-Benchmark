import type { ReactNode } from 'react'

/** A failed request: what failed, and a way to try again. */
export function QueryError({
  children,
  onRetry,
  retrying,
}: {
  children: ReactNode
  onRetry: () => void
  retrying: boolean
}) {
  return (
    <div className="status status-error" role="alert">
      <p>{children}</p>
      <button type="button" onClick={onRetry} disabled={retrying}>
        {retrying ? 'Retrying…' : 'Retry'}
      </button>
    </div>
  )
}

/** The tracking database has nothing in it yet; says how to fill it. */
export function EmptyDatabase({ nothing }: { nothing: string }) {
  return (
    <div className="status" role="status">
      <p>
        No {nothing} are recorded yet. From the repository root, run{' '}
        <code>python -m src.backfill --experiment-dir runs/multiseed/experiment1</code>, then reload.
      </p>
    </div>
  )
}
