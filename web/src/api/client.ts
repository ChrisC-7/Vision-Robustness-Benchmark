import createClient from 'openapi-fetch'
import type { paths } from './schema'

export const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// Every path, parameter and response body below is checked against
// schema.d.ts, which `npm run gen:api` generates from the backend's OpenAPI
// schema. Rename a field in src/api.py and this stops compiling.
export const api = createClient<paths>({
  baseUrl: apiBaseUrl,
  // openapi-fetch otherwise captures globalThis.fetch once, at import time.
  // Looking it up per request is what lets tests stub it with vi.stubGlobal.
  fetch: (request) => globalThis.fetch(request),
})
