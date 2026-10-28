import { Link } from 'react-router'
import { useRuns } from '../api/runs'
import { EmptyDatabase, Loading, QueryError } from '../components/QueryStates'
import { MODEL_LABELS, seedLabel } from '../results/evaluations'
import { formatAccuracy } from '../results/format'
import { experimentName, runPath } from './runPath'

export function RunsPage() {
  const runs = useRuns()

  if (runs.isPending) {
    return <Loading>Loading runs…</Loading>
  }

  if (runs.isError) {
    return (
      <QueryError onRetry={() => void runs.refetch()} retrying={runs.isFetching}>
        Couldn't load runs: {runs.error.message}
      </QueryError>
    )
  }

  if (runs.data.length === 0) {
    return <EmptyDatabase nothing="training runs" />
  }

  return (
    <div className="table-scroll">
      <table className="data-table">
        <caption>Recorded training runs. Open one for its full training config.</caption>
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Experiment</th>
            <th scope="col">Model</th>
            <th scope="col" className="num">
              Training seed
            </th>
            <th scope="col" className="num">
              Split seed
            </th>
            <th scope="col" className="num">
              Best epoch
            </th>
            <th scope="col" className="num">
              Best val. accuracy (%)
            </th>
          </tr>
        </thead>
        <tbody>
          {runs.data.map((run) => (
            <tr key={run.run_dir}>
              <th scope="row">
                <Link to={runPath(run.run_dir)}>{run.run_name}</Link>
              </th>
              <td>{experimentName(run.run_dir) ?? '—'}</td>
              <td>{MODEL_LABELS[run.model_name]}</td>
              <td className="num">{seedLabel(run.training_seed)}</td>
              <td className="num">{seedLabel(run.split_seed)}</td>
              <td className="num">{run.best_epoch ?? '—'}</td>
              <td className="num">{formatAccuracy(run.best_val_accuracy ?? undefined)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
