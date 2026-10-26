import { useQuery } from '@tanstack/react-query'
import { groupEvaluations, type EvalResult } from '../results/evaluations'
import { api, ApiError } from './client'

export async function fetchResults(): Promise<EvalResult[]> {
  const { data, response } = await api.GET('/results')
  if (data === undefined) throw new ApiError('GET /results', response.status)
  return data
}

/**
 * Every recorded evaluation, fetched once and filtered in the browser: the
 * whole table is a few dozen rows, and each Δ needs clean and corrupted rows
 * together, which a server-side condition filter would split apart.
 */
export function useEvaluations() {
  // select runs only when the fetched rows change, so grouping is not redone
  // on every render.
  return useQuery({ queryKey: ['results'], queryFn: fetchResults, select: groupEvaluations })
}
