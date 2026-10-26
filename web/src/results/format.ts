import type { Summary } from './evaluations'

const twoDecimals = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})
const signedTwoDecimals = new Intl.NumberFormat('en-US', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
  signDisplay: 'exceptZero',
})

const MISSING = '—'

export function formatAccuracy(value: number | undefined): string {
  return value === undefined ? MISSING : twoDecimals.format(value)
}

export function formatDrop(value: number | undefined): string {
  return value === undefined ? MISSING : signedTwoDecimals.format(value)
}

/** "38.01 ± 3.54", as in the README tables; just the mean when there is one value. */
export function formatSummary(summary: Summary | undefined, format: (value: number) => string): string {
  if (summary === undefined) return MISSING
  if (summary.sd === undefined) return format(summary.mean)
  return `${format(summary.mean)} ± ${twoDecimals.format(summary.sd)}`
}
