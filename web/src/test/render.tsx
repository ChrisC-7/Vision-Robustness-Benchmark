import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { vi } from 'vitest'

type TestRoute = { path: string; element: ReactNode }

/**
 * Renders pages the way the app does: inside a fresh QueryClient and a
 * router at `url`, with fetch answering every request from `respond`. Any
 * URL no route claims renders "Navigated away". Retries happen without
 * delay, so a test that hits one doesn't wait a second for it.
 */
export function renderPage(routes: TestRoute[], url: string, respond: (request: Request) => Response) {
  const fetchMock = vi.fn(async (request: Request) => respond(request))
  vi.stubGlobal('fetch', fetchMock)
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false, retryDelay: 0 } } })
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[url]}>
        <Routes>
          {routes.map(({ path, element }) => (
            <Route key={path} path={path} element={element} />
          ))}
          <Route path="*" element={<p>Navigated away</p>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
  return { fetchMock, user: userEvent.setup() }
}
