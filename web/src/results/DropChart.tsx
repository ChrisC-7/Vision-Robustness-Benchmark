import { Bar, BarChart, CartesianGrid, ReferenceLine, Tooltip, XAxis, YAxis } from 'recharts'
import { CONDITION_LABELS, MODEL_LABELS, seedLabel, type Corruption, type DropRow, type ModelName } from './evaluations'
import { formatDrop } from './format'
import { AXIS_TICK, BAR_PROPS, CHART_MARGIN, SERIES_COLORS } from './chartStyle'
import { ChartTooltip, type TooltipItem } from './series'

type Props = {
  corruption: Corruption
  rows: readonly DropRow[]
  models: readonly ModelName[]
  /** Shared with the other drop chart, so equal bar lengths mean equal drops. */
  domain: [number, number]
}

export function DropChart({ corruption, rows, models, domain }: Props) {
  return (
    <BarChart
      responsive
      data={rows as DropRow[]}
      style={{ width: '100%', height: 220 }}
      margin={CHART_MARGIN}
      barGap={2}
      barCategoryGap="30%"
    >
      <CartesianGrid vertical={false} stroke="var(--grid)" />
      <XAxis
        dataKey={(row: DropRow) => seedLabel(row.seed)}
        tick={AXIS_TICK}
        tickLine={false}
        axisLine={false}
      />
      <YAxis domain={domain} unit=" pp" width={56} tick={AXIS_TICK} tickLine={false} axisLine={false} />
      <ReferenceLine y={0} stroke="var(--axis)" />
      <Tooltip
        cursor={{ fill: 'var(--hover-wash)' }}
        content={({ active, payload }) => {
          const row = payload?.[0]?.payload as DropRow | undefined
          if (!active || row === undefined) return null
          return (
            <ChartTooltip
              title={`${CONDITION_LABELS[corruption]} Δ, seed ${seedLabel(row.seed)}`}
              items={models.flatMap((model): TooltipItem[] => {
                const summary = row.summaries[model]
                return summary === undefined ? [] : [{ model, value: `${formatDrop(summary.mean)} pp` }]
              })}
            />
          )
        }}
      />
      {models.map((model) => (
        <Bar
          key={model}
          name={MODEL_LABELS[model]}
          dataKey={(row: DropRow) => row.summaries[model]?.mean}
          fill={SERIES_COLORS[model]}
          radius={[4, 4, 0, 0]}
          {...BAR_PROPS}
        />
      ))}
    </BarChart>
  )
}
