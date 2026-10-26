// Shared Recharts settings so every chart on the page uses the same marks.

import type { ModelName } from './evaluations'

// Color follows the model, never its position among the visible series:
// hiding SimpleCNN must not repaint StrongerCNN. Values are palette slots 1
// and 2, defined per theme in index.css.
export const SERIES_COLORS: Record<ModelName, string> = {
  simple: 'var(--series-1)',
  stronger: 'var(--series-2)',
}

export const CHART_MARGIN = { top: 8, right: 8, bottom: 0, left: 0 }

export const AXIS_TICK = { fill: 'var(--text-muted)', fontSize: 12 }

export const BAR_PROPS = {
  // Bars stay thin and leave the rest of their band as air.
  maxBarSize: 24,
  // Off so every render (and every test or screenshot) shows final values.
  isAnimationActive: false,
} as const
