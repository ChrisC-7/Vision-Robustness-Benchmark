import type { components } from '../api/schema'

export type EvalResult = components['schemas']['EvalResult']
export type ModelName = EvalResult['model_name']
export type Condition = EvalResult['condition']
export type Corruption = Exclude<Condition, 'clean'>

// Records keyed by the generated unions: if the backend adds a model or a
// condition, these stop compiling until it gets a label.
export const MODEL_LABELS: Record<ModelName, string> = {
  simple: 'SimpleCNN',
  stronger: 'StrongerCNN',
}
export const CONDITION_LABELS: Record<Condition, string> = {
  clean: 'Clean',
  blur: 'Blur',
  noise: 'Noise',
}
export const MODELS = Object.keys(MODEL_LABELS) as ModelName[]
export const CORRUPTIONS: readonly Corruption[] = ['blur', 'noise']

/** One evaluation attempt of one trained run, with every condition it measured. */
export type Evaluation = {
  evalId: string
  runName: string
  model: ModelName
  trainingSeed: number | null
  accuracy: Partial<Record<Condition, number>>
}

/**
 * Turns the API's one-row-per-condition results into one entry per
 * evaluation attempt, sorted by model, then training seed.
 *
 * Grouping by eval_id rather than by run is what keeps every Δ honest: clean
 * and corrupted accuracy always come from the same attempt, even if a run
 * was evaluated more than once.
 */
export function groupEvaluations(rows: readonly EvalResult[]): Evaluation[] {
  const byId = new Map<string, Evaluation>()
  for (const row of rows) {
    let evaluation = byId.get(row.eval_id)
    if (evaluation === undefined) {
      evaluation = {
        evalId: row.eval_id,
        runName: row.run_name,
        model: row.model_name,
        trainingSeed: row.training_seed,
        accuracy: {},
      }
      byId.set(row.eval_id, evaluation)
    }
    evaluation.accuracy[row.condition] = row.accuracy
  }
  return [...byId.values()].sort(
    (a, b) =>
      MODELS.indexOf(a.model) - MODELS.indexOf(b.model) ||
      compareSeeds(a.trainingSeed, b.trainingSeed) ||
      a.evalId.localeCompare(b.evalId),
  )
}

/**
 * Corrupted minus clean accuracy, in percentage points, so a drop is
 * negative. Undefined unless the attempt measured both conditions.
 */
export function accuracyDrop(evaluation: Evaluation, corruption: Corruption): number | undefined {
  const clean = evaluation.accuracy.clean
  const corrupted = evaluation.accuracy[corruption]
  return clean === undefined || corrupted === undefined ? undefined : corrupted - clean
}

export type Summary = { mean: number; sd: number | undefined; n: number }

/**
 * Mean and sample standard deviation (n - 1), matching the README tables.
 * sd is undefined for a single value; the whole summary for none.
 */
export function summarize(values: readonly number[]): Summary | undefined {
  const n = values.length
  if (n === 0) return undefined
  const mean = values.reduce((sum, value) => sum + value, 0) / n
  if (n === 1) return { mean, sd: undefined, n }
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (n - 1)
  return { mean, sd: Math.sqrt(variance), n }
}

export type ModelSummaries = Partial<Record<ModelName, Summary>>

export function summarizeByModel(
  evaluations: readonly Evaluation[],
  models: readonly ModelName[],
  value: (evaluation: Evaluation) => number | undefined,
): ModelSummaries {
  const summaries: ModelSummaries = {}
  for (const model of models) {
    const values = evaluations
      .filter((evaluation) => evaluation.model === model)
      .map(value)
      .filter((v) => v !== undefined)
    const summary = summarize(values)
    if (summary !== undefined) summaries[model] = summary
  }
  return summaries
}

export type AccuracyRow = { condition: Condition; summaries: ModelSummaries }

/** One row per condition: each model's accuracy, summarized over its evaluations. */
export function accuracyRows(
  evaluations: readonly Evaluation[],
  models: readonly ModelName[],
  conditions: readonly Condition[],
): AccuracyRow[] {
  return conditions.map((condition) => ({
    condition,
    summaries: summarizeByModel(evaluations, models, (evaluation) => evaluation.accuracy[condition]),
  }))
}

export type DropRow = { seed: number | null; summaries: ModelSummaries }

/** One row per training seed: each model's drop under one corruption. */
export function dropRows(
  evaluations: readonly Evaluation[],
  models: readonly ModelName[],
  corruption: Corruption,
): DropRow[] {
  return uniqueSeeds(evaluations).map((seed) => ({
    seed,
    summaries: summarizeByModel(
      evaluations.filter((evaluation) => evaluation.trainingSeed === seed),
      models,
      (evaluation) => accuracyDrop(evaluation, corruption),
    ),
  }))
}

/**
 * A y-domain that includes 0 and every mean, widened to multiples of 5, so
 * that side-by-side drop charts share one scale and bars compare across them.
 */
export function sharedDomain(rowSets: readonly (readonly DropRow[])[]): [number, number] {
  const means = rowSets.flatMap((rows) =>
    rows.flatMap((row) => Object.values(row.summaries).map((summary) => summary.mean)),
  )
  const low = Math.min(0, ...means)
  const high = Math.max(0, ...means)
  return [Math.floor(low / 5) * 5, Math.ceil(high / 5) * 5]
}

export function uniqueSeeds(evaluations: readonly Evaluation[]): (number | null)[] {
  return [...new Set(evaluations.map((evaluation) => evaluation.trainingSeed))].sort(compareSeeds)
}

export function seedLabel(seed: number | null): string {
  return seed === null ? 'unknown' : String(seed)
}

/** Ascending, with an unrecorded (null) seed last. */
function compareSeeds(a: number | null, b: number | null): number {
  if (a === b) return 0
  if (a === null) return 1
  if (b === null) return -1
  return a - b
}
