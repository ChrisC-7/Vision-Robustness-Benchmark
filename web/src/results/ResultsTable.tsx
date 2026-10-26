import {
  CONDITION_LABELS,
  MODEL_LABELS,
  accuracyDrop,
  seedLabel,
  summarizeByModel,
  type Condition,
  type Corruption,
  type Evaluation,
  type ModelName,
} from './evaluations'
import { formatAccuracy, formatDrop, formatSummary } from './format'

type Props = {
  evaluations: readonly Evaluation[]
  models: readonly ModelName[]
  corruptions: readonly Corruption[]
}

/** The charts' table twin: every value they plot, readable without hovering. */
export function ResultsTable({ evaluations, models, corruptions }: Props) {
  const conditions: Condition[] = ['clean', ...corruptions]

  // The note sits outside the scroll container so it wraps to the screen
  // instead of to the table's width, where a narrow screen would cut it off.
  return (
    <>
      <div className="table-scroll">
        <table className="data-table" aria-describedby="results-table-note">
          <caption>Per-run results</caption>
          <thead>
            <tr>
              <th scope="col">Model</th>
              <th scope="col">Seed</th>
              {conditions.map((condition) => (
                <th scope="col" className="num" key={condition}>
                  {CONDITION_LABELS[condition]}
                </th>
              ))}
              {corruptions.map((corruption) => (
                <th scope="col" className="num" key={`${corruption}-drop`}>
                  {CONDITION_LABELS[corruption]} Δ
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {evaluations.map((evaluation) => (
              <tr key={evaluation.evalId}>
                <th scope="row">{MODEL_LABELS[evaluation.model]}</th>
                <td>{seedLabel(evaluation.trainingSeed)}</td>
                {conditions.map((condition) => (
                  <td className="num" key={condition}>
                    {formatAccuracy(evaluation.accuracy[condition])}
                  </td>
                ))}
                {corruptions.map((corruption) => (
                  <td className="num" key={`${corruption}-drop`}>
                    {formatDrop(accuracyDrop(evaluation, corruption))}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
          <tfoot>
            {models.map((model) => {
              const accuracy = (condition: Condition) =>
                summarizeByModel(evaluations, [model], (e) => e.accuracy[condition])[model]
              const drop = (corruption: Corruption) =>
                summarizeByModel(evaluations, [model], (e) => accuracyDrop(e, corruption))[model]
              if (accuracy('clean') === undefined) return null
              return (
                <tr key={model}>
                  <th scope="row">{MODEL_LABELS[model]}</th>
                  <td>Mean</td>
                  {conditions.map((condition) => (
                    <td className="num" key={condition}>
                      {formatSummary(accuracy(condition), formatAccuracy)}
                    </td>
                  ))}
                  {corruptions.map((corruption) => (
                    <td className="num" key={`${corruption}-drop`}>
                      {formatSummary(drop(corruption), formatDrop)}
                    </td>
                  ))}
                </tr>
              )
            })}
          </tfoot>
        </table>
      </div>
      <p id="results-table-note" className="table-note">
        Accuracy in %, Δ in percentage points (corrupted minus clean). Mean rows are mean ± sample standard deviation
        over the selected seeds.
      </p>
    </>
  )
}
