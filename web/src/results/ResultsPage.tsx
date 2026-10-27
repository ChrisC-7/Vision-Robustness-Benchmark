import { useState } from 'react'
import { useEvaluations } from '../api/results'
import { EmptyDatabase, Loading, QueryError } from '../components/QueryStates'
import { AccuracyChart } from './AccuracyChart'
import { DropChart } from './DropChart'
import {
  CONDITION_LABELS,
  CORRUPTIONS,
  MODELS,
  accuracyRows,
  dropRows,
  sharedDomain,
  uniqueSeeds,
  type Evaluation,
} from './evaluations'
import { ResultsFilters, type HiddenSets } from './ResultsFilters'
import { ResultsTable } from './ResultsTable'
import { ModelLegend } from './series'

export function ResultsPage() {
  const results = useEvaluations()

  if (results.isPending) {
    return <Loading>Loading results…</Loading>
  }

  if (results.isError) {
    return (
      <QueryError onRetry={() => void results.refetch()} retrying={results.isFetching}>
        Couldn't load results: {results.error.message}
      </QueryError>
    )
  }

  if (results.data.length === 0) {
    return <EmptyDatabase nothing="evaluation results" />
  }

  return <ResultsView evaluations={results.data} />
}

const NOTHING_HIDDEN: HiddenSets = { models: new Set(), seeds: new Set(), corruptions: new Set() }

function ResultsView({ evaluations }: { evaluations: readonly Evaluation[] }) {
  const [hidden, setHidden] = useState<HiddenSets>(NOTHING_HIDDEN)

  const visible = evaluations.filter(
    (evaluation) => !hidden.models.has(evaluation.model) && !hidden.seeds.has(evaluation.trainingSeed),
  )
  const models = MODELS.filter((model) => visible.some((evaluation) => evaluation.model === model))
  const corruptions = CORRUPTIONS.filter((corruption) => !hidden.corruptions.has(corruption))
  const drops = corruptions.map((corruption) => ({ corruption, rows: dropRows(visible, models, corruption) }))
  const dropDomain = sharedDomain(drops.map((drop) => drop.rows))

  return (
    <>
      <ResultsFilters seeds={uniqueSeeds(evaluations)} hidden={hidden} onChange={setHidden} />

      {visible.length === 0 ? (
        <p className="status" role="status">
          No runs match these filters.
        </p>
      ) : (
        <>
          <ModelLegend models={models} />

          <figure className="chart-card">
            <figcaption>
              <h3>Accuracy by condition</h3>
              <p>Mean over the selected seeds. Hover a bar for the spread.</p>
            </figcaption>
            <AccuracyChart rows={accuracyRows(visible, models, ['clean', ...corruptions])} models={models} />
          </figure>

          {drops.length > 0 && (
            <div className="chart-grid">
              {drops.map(({ corruption, rows }) => (
                <figure className="chart-card" key={corruption}>
                  <figcaption>
                    <h3>{CONDITION_LABELS[corruption]} Δ by training seed</h3>
                    <p>Accuracy change from clean, in percentage points.</p>
                  </figcaption>
                  <DropChart corruption={corruption} rows={rows} models={models} domain={dropDomain} />
                </figure>
              ))}
            </div>
          )}

          <ResultsTable evaluations={visible} models={models} corruptions={corruptions} />
        </>
      )}
    </>
  )
}
