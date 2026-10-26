import { SERIES_COLORS } from './chartStyle'
import { MODEL_LABELS, type ModelName } from './evaluations'

export function ModelLegend({ models }: { models: readonly ModelName[] }) {
  return (
    <ul className="legend" aria-label="Models">
      {models.map((model) => (
        <li key={model}>
          <span className="legend-swatch" style={{ background: SERIES_COLORS[model] }} aria-hidden="true" />
          {MODEL_LABELS[model]}
        </li>
      ))}
    </ul>
  )
}

export type TooltipItem = { model: ModelName; value: string; detail?: string }

/** Values lead and model names follow; each row is keyed by a short stroke in the series color. */
export function ChartTooltip({ title, items }: { title: string; items: readonly TooltipItem[] }) {
  return (
    <div className="chart-tooltip">
      <p className="chart-tooltip-title">{title}</p>
      {items.map((item) => (
        <p key={item.model} className="chart-tooltip-row">
          <span className="chart-tooltip-key" style={{ background: SERIES_COLORS[item.model] }} aria-hidden="true" />
          <strong>{item.value}</strong> {MODEL_LABELS[item.model]}
          {item.detail !== undefined && <span className="chart-tooltip-detail"> {item.detail}</span>}
        </p>
      ))}
    </div>
  )
}
