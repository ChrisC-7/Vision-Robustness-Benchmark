import { useEffect, useState, type ReactNode } from 'react'

/** How long a request may take before it gets the cold-start explanation. */
export const SLOW_REQUEST_MS = 4000

/**
 * A request in flight. The deployed API runs on a free instance that sleeps
 * when idle, so a request that is still pending after a few seconds says so
 * instead of looking stuck.
 */
export function Loading({ children }: { children: ReactNode }) {
  const [slow, setSlow] = useState(false)
  useEffect(() => {
    const timeout = setTimeout(() => setSlow(true), SLOW_REQUEST_MS)
    return () => clearTimeout(timeout)
  }, [])

  return (
    <div className="status" role="status">
      <p>{children}</p>
      {slow && (
        <p className="hint">The API runs on a free server that sleeps when idle; waking it can take up to a minute.</p>
      )}
    </div>
  )
}

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
