import { useQuery } from '@tanstack/react-query'
import { api, ApiError } from './client'
import type { components } from './schema'

export type Health = components['schemas']['HealthResponse']

export async function fetchHealth(): Promise<Health> {
  const { data, response } = await api.GET('/health')
  // openapi-fetch never throws on an HTTP error; it leaves data undefined.
  // Throwing here is what puts TanStack Query into its error state.
  if (data === undefined) throw new ApiError('GET /health', response.status)
  return data
}

export function useHealth() {
  return useQuery({ queryKey: ['health'], queryFn: fetchHealth })
}
