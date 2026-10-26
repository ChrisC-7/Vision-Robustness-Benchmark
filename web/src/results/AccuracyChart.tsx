import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from 'recharts'
import { CONDITION_LABELS, MODEL_LABELS, type AccuracyRow, type ModelName, type Summary } from './evaluations'
import { formatAccuracy } from './format'
import { AXIS_TICK, BAR_PROPS, CHART_MARGIN, SERIES_COLORS } from './chartStyle'
import { ChartTooltip, type TooltipItem } from './series'

type Props = { rows: readonly AccuracyRow[]; models: readonly ModelName[] }

export function AccuracyChart({ rows, models }: Props) {
  return (
    <BarChart
      responsive
      data={rows as AccuracyRow[]}
      style={{ width: '100%', height: 260 }}
      margin={CHART_MARGIN}
      barGap={2}
      barCategoryGap="30%"
    >
      <CartesianGrid vertical={false} stroke="var(--grid)" />
      <XAxis
        dataKey={(row: AccuracyRow) => CONDITION_LABELS[row.condition]}
        tick={AXIS_TICK}
        tickLine={false}
        axisLine={{ stroke: 'var(--axis)' }}
      />
      <YAxis
        domain={[0, 100]}
        ticks={[0, 25, 50, 75, 100]}
        unit="%"
        width={44}
        tick={AXIS_TICK}
        tickLine={false}
        axisLine={false}
      />
      <Tooltip
        cursor={{ fill: 'var(--hover-wash)' }}
        content={({ active, payload }) => {
          const row = payload?.[0]?.payload as AccuracyRow | undefined
          if (!active || row === undefined) return null
          return (
            <ChartTooltip
              title={`${CONDITION_LABELS[row.condition]} accuracy`}
              items={tooltipItems(row.summaries, models)}
            />
          )
        }}
      />
      {models.map((model) => (
        <Bar
          key={model}
          name={MODEL_LABELS[model]}
          dataKey={(row: AccuracyRow) => row.summaries[model]?.mean}
          fill={SERIES_COLORS[model]}
          radius={[4, 4, 0, 0]}
          {...BAR_PROPS}
        />
      ))}
    </BarChart>
  )
}

function tooltipItems(summaries: Partial<Record<ModelName, Summary>>, models: readonly ModelName[]): TooltipItem[] {
  return models.flatMap((model) => {
    const summary = summaries[model]
    if (summary === undefined) return []
    const seeds = `${summary.n} ${summary.n === 1 ? 'seed' : 'seeds'}`
    return [{
      model,
      value: `${formatAccuracy(summary.mean)}%`,
      detail: summary.sd === undefined ? seeds : `± ${formatAccuracy(summary.sd)} · ${seeds}`,
    }]
  })
}
