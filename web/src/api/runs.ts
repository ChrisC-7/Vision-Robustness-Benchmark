import { useQuery } from '@tanstack/react-query'
import { api, ApiError } from './client'
import type { components } from './schema'

export type RunSummary = components['schemas']['RunSummary']
export type RunDetail = components['schemas']['RunDetail']

export async function fetchRuns(): Promise<RunSummary[]> {
  const { data, response } = await api.GET('/runs')
  if (data === undefined) throw new ApiError('GET /runs', response.status)
  return data
}

export async function fetchRun(runDir: string): Promise<RunDetail> {
  // openapi-fetch percent-encodes path parameters, so runDir's slashes
  // travel as %2F and the API's {run_dir:path} route decodes them back.
  const { data, response } = await api.GET('/runs/{run_dir}', { params: { path: { run_dir: runDir } } })
  if (data === undefined) throw new ApiError(`GET /runs/${runDir}`, response.status)
  return data
}

export function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}

export function useRuns() {
  return useQuery({ queryKey: ['runs'], queryFn: fetchRuns })
}

export function useRun(runDir: string) {
  return useQuery({
    queryKey: ['runs', runDir],
    queryFn: () => fetchRun(runDir),
    // Retry an outage once, like every other query, but not a 404: a run
    // that isn't recorded won't be on the second try either.
    retry: (failureCount, error) => !isNotFound(error) && failureCount < 1,
  })
}
