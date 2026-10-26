import { apiBaseUrl } from '../api/client'
import { useHealth } from '../api/health'
import { QueryError } from './QueryStates'

export function HealthStatus() {
  const health = useHealth()

  if (health.isPending) {
    return (
      <p className="status" role="status">
        Checking the API at <code>{apiBaseUrl}</code>…
      </p>
    )
  }

  if (health.isError) {
    return (
      <QueryError onRetry={() => void health.refetch()} retrying={health.isFetching}>
        Can't reach the API at <code>{apiBaseUrl}</code>: {health.error.message}
      </QueryError>
    )
  }

  // isPending and isError are both ruled out, so TypeScript knows data is defined here.
  const { models } = health.data
  return (
    <div className="status status-ok" role="status">
      <p>
        API online at <code>{apiBaseUrl}</code> with {models.length} models loaded:
      </p>
      <ul>
        {models.map((model) => (
          <li key={model}>
            <code>{model}</code>
          </li>
        ))}
      </ul>
    </div>
  )
}
