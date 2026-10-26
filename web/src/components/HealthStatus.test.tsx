import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import { apiBaseUrl } from '../api/client'
import { HealthStatus } from './HealthStatus'

function renderHealthStatus() {
  // A fresh client per test so no cached response carries over, with retries
  // off so a failing request reaches the error state immediately.
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <HealthStatus />
    </QueryClientProvider>,
  )
}

it('lists the loaded models when the API is up', async () => {
  const fetchMock = vi.fn(async (_request: Request) =>
    Response.json({ status: 'ok', models: ['simple', 'stronger'] }),
  )
  vi.stubGlobal('fetch', fetchMock)

  renderHealthStatus()

  expect(await screen.findByText(/with 2 models loaded/)).toBeInTheDocument()
  expect(screen.getByText('simple')).toBeInTheDocument()
  expect(screen.getByText('stronger')).toBeInTheDocument()
  expect(fetchMock.mock.calls[0]?.[0].url).toBe(`${apiBaseUrl}/health`)
})

it('shows the API address and the network error when the API is down', async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      throw new TypeError('Failed to fetch')
    }),
  )

  renderHealthStatus()

  const alert = await screen.findByRole('alert')
  expect(alert).toHaveTextContent(apiBaseUrl)
  expect(alert).toHaveTextContent('Failed to fetch')
  expect(screen.getByRole('button', { name: 'Retry' })).toBeEnabled()
})

it('reports the HTTP status when the API answers with an error', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response('Internal Server Error', { status: 500 })))

  renderHealthStatus()

  expect(await screen.findByRole('alert')).toHaveTextContent('GET /health failed with HTTP 500')
})
