import { CONDITION_LABELS, CORRUPTIONS, MODELS, MODEL_LABELS, seedLabel, type Corruption, type ModelName } from './evaluations'

/**
 * Filters store what is *hidden*, not what is shown, so a seed or model that
 * appears in a later refetch is visible by default instead of silently
 * filtered out.
 */
export type HiddenSets = {
  models: ReadonlySet<ModelName>
  seeds: ReadonlySet<number | null>
  corruptions: ReadonlySet<Corruption>
}

type Props = {
  seeds: readonly (number | null)[]
  hidden: HiddenSets
  onChange: (hidden: HiddenSets) => void
}

/** One row of filters above everything they scope. */
export function ResultsFilters({ seeds, hidden, onChange }: Props) {
  return (
    <div className="filters">
      <CheckboxGroup
        legend="Model"
        options={MODELS}
        label={(model) => MODEL_LABELS[model]}
        hidden={hidden.models}
        onChange={(models) => onChange({ ...hidden, models })}
      />
      <CheckboxGroup
        legend="Training seed"
        options={seeds}
        label={seedLabel}
        hidden={hidden.seeds}
        onChange={(next) => onChange({ ...hidden, seeds: next })}
      />
      <CheckboxGroup
        legend="Corruption"
        options={CORRUPTIONS}
        label={(corruption) => CONDITION_LABELS[corruption]}
        hidden={hidden.corruptions}
        onChange={(corruptions) => onChange({ ...hidden, corruptions })}
      />
    </div>
  )
}

type CheckboxGroupProps<T> = {
  legend: string
  options: readonly T[]
  label: (option: T) => string
  hidden: ReadonlySet<T>
  onChange: (hidden: ReadonlySet<T>) => void
}

function CheckboxGroup<T>({ legend, options, label, hidden, onChange }: CheckboxGroupProps<T>) {
  return (
    <fieldset className="filter-group">
      <legend>{legend}</legend>
      {options.map((option) => (
        <label key={label(option)}>
          <input type="checkbox" checked={!hidden.has(option)} onChange={() => onChange(toggle(hidden, option))} />
          {label(option)}
        </label>
      ))}
    </fieldset>
  )
}

function toggle<T>(set: ReadonlySet<T>, value: T): ReadonlySet<T> {
  const next = new Set(set)
  if (!next.delete(value)) next.add(value)
  return next
}
