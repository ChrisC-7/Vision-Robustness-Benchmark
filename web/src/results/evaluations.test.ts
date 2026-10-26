import { describe, expect, it } from 'vitest'
import {
  accuracyDrop,
  accuracyRows,
  dropRows,
  groupEvaluations,
  sharedDomain,
  summarize,
  type Condition,
  type EvalResult,
  type ModelName,
} from './evaluations'
import { formatAccuracy, formatDrop, formatSummary } from './format'

function row(
  model: ModelName,
  seed: number | null,
  condition: Condition,
  accuracy: number,
  evalId = `${model}_s${seed}`,
): EvalResult {
  return {
    run_dir: `runs/multiseed/experiment1/${model}_s${seed}`,
    run_name: `${model}_s${seed}`,
    model_name: model,
    training_seed: seed,
    eval_id: evalId,
    noise_seed: 42,
    condition,
    loss: 1,
    accuracy,
    params: {},
  }
}

describe('groupEvaluations', () => {
  it('collects one attempt per eval_id, sorted by model then seed', () => {
    const evaluations = groupEvaluations([
      row('stronger', 42, 'clean', 70),
      row('simple', 123, 'clean', 35),
      row('simple', 42, 'blur', 30),
      row('simple', 42, 'clean', 40),
    ])

    expect(evaluations.map((e) => e.evalId)).toEqual(['simple_s42', 'simple_s123', 'stronger_s42'])
    expect(evaluations[0]?.accuracy).toEqual({ clean: 40, blur: 30 })
  })

  it('keeps two attempts of the same run apart', () => {
    const evaluations = groupEvaluations([
      row('simple', 42, 'clean', 40, 'first'),
      row('simple', 42, 'clean', 41, 'second'),
    ])

    expect(evaluations).toHaveLength(2)
  })

  it('puts a run with no recorded seed last', () => {
    const evaluations = groupEvaluations([row('simple', null, 'clean', 40), row('simple', 7, 'clean', 40)])

    expect(evaluations.map((e) => e.trainingSeed)).toEqual([7, null])
  })
})

describe('accuracyDrop', () => {
  it('is corrupted minus clean, so a loss is negative', () => {
    const [evaluation] = groupEvaluations([row('simple', 42, 'clean', 41.06), row('simple', 42, 'blur', 28.49)])

    expect(accuracyDrop(evaluation!, 'blur')).toBeCloseTo(-12.57)
  })

  it('is undefined when the attempt never measured clean', () => {
    const [evaluation] = groupEvaluations([row('simple', 42, 'noise', 30)])

    expect(accuracyDrop(evaluation!, 'noise')).toBeUndefined()
  })
})

describe('summarize', () => {
  it('uses the sample standard deviation, like the README tables', () => {
    const summary = summarize([2, 4, 4, 4, 5, 5, 7, 9])

    expect(summary?.mean).toBe(5)
    expect(summary?.sd).toBeCloseTo(2.138, 3)
  })

  it('has no spread for a single value and no summary for none', () => {
    expect(summarize([40])).toEqual({ mean: 40, sd: undefined, n: 1 })
    expect(summarize([])).toBeUndefined()
  })
})

describe('chart rows', () => {
  const evaluations = groupEvaluations([
    row('simple', 42, 'clean', 40),
    row('simple', 42, 'noise', 30),
    row('simple', 123, 'clean', 36),
    row('simple', 123, 'noise', 30),
    row('stronger', 42, 'clean', 70),
    row('stronger', 42, 'noise', 62),
  ])

  it('summarizes accuracy per condition and model', () => {
    const [clean] = accuracyRows(evaluations, ['simple', 'stronger'], ['clean'])

    expect(clean?.summaries.simple?.mean).toBe(38)
    expect(clean?.summaries.stronger).toEqual({ mean: 70, sd: undefined, n: 1 })
  })

  it('gives one drop row per seed, leaving out models that lack that seed', () => {
    const rows = dropRows(evaluations, ['simple', 'stronger'], 'noise')

    expect(rows.map((r) => r.seed)).toEqual([42, 123])
    expect(rows[0]?.summaries).toEqual({
      simple: { mean: -10, sd: undefined, n: 1 },
      stronger: { mean: -8, sd: undefined, n: 1 },
    })
    expect(rows[1]?.summaries.stronger).toBeUndefined()
  })

  it('shares one domain across drop charts, rounded out to multiples of 5', () => {
    const noise = dropRows(evaluations, ['simple'], 'noise')

    expect(sharedDomain([noise, [{ seed: 1, summaries: { simple: { mean: -17.2, sd: undefined, n: 1 } } }]])).toEqual([
      -20, 0,
    ])
  })
})

describe('format', () => {
  it('signs drops and shows mean ± SD', () => {
    expect(formatDrop(-9.04)).toBe('-9.04')
    expect(formatDrop(1.5)).toBe('+1.50')
    expect(formatSummary(summarize([40, 36]), formatAccuracy)).toBe('38.00 ± 2.83')
    expect(formatSummary(summarize([-9]), formatDrop)).toBe('-9.00')
  })
})
