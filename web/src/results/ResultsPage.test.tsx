import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, it, vi } from 'vitest'
import type { EvalResult, ModelName } from './evaluations'
import { ResultsPage } from './ResultsPage'

// [model, seed, clean, blur, noise]
const RUNS: [ModelName, number, number, number, number][] = [
  ['simple', 42, 40, 30, 32],
  ['simple', 123, 36, 28, 26],
  ['stronger', 42, 70, 52, 61],
  ['stronger', 123, 66, 50, 58],
]

const RESULTS: EvalResult[] = RUNS.flatMap(([model, seed, clean, blur, noise]) =>
  ([['clean', clean], ['blur', blur], ['noise', noise]] as const).map(([condition, accuracy]) => ({
    run_dir: `runs/multiseed/experiment1/${model}_s${seed}`,
    run_name: `${model}_s${seed}`,
    model_name: model,
    training_seed: seed,
    eval_id: `backfill:${model}_s${seed}`,
    noise_seed: 42,
    condition,
    loss: 1,
    accuracy,
    params: {},
  })),
)

function renderResultsPage(response: () => Response = () => Response.json(RESULTS)) {
  vi.stubGlobal('fetch', vi.fn(async () => response()))
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <ResultsPage />
    </QueryClientProvider>,
  )
  return userEvent.setup()
}

it('shows one row per run and a mean ± SD row per model', async () => {
  renderResultsPage()

  expect(await screen.findByRole('row', { name: /^SimpleCNN 42 40\.00 30\.00 32\.00 -10\.00 -8\.00$/ })).toBeInTheDocument()
  expect(screen.getByRole('row', { name: /^StrongerCNN 123 / })).toBeInTheDocument()
  expect(
    screen.getByRole('row', {
      name: 'SimpleCNN Mean 38.00 ± 2.83 29.00 ± 1.41 29.00 ± 4.24 -9.00 ± 1.41 -9.00 ± 1.41',
    }),
  ).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Blur Δ by training seed' })).toBeInTheDocument()
  expect(screen.getByRole('heading', { name: 'Noise Δ by training seed' })).toBeInTheDocument()
})

it('hides a model everywhere when it is unchecked', async () => {
  const user = renderResultsPage()

  await user.click(await screen.findByRole('checkbox', { name: 'StrongerCNN' }))

  expect(screen.queryByRole('row', { name: /^StrongerCNN/ })).not.toBeInTheDocument()
  expect(screen.getByRole('row', { name: /^SimpleCNN Mean/ })).toBeInTheDocument()
  expect(screen.getByRole('list', { name: 'Models' })).toHaveTextContent(/^SimpleCNN$/)
})

it('recomputes the mean rows over the seeds left checked', async () => {
  const user = renderResultsPage()

  await user.click(await screen.findByRole('checkbox', { name: '123' }))

  expect(screen.queryByRole('row', { name: /^SimpleCNN 123/ })).not.toBeInTheDocument()
  // A single seed has no spread, so the mean row shows just the value.
  expect(
    screen.getByRole('row', { name: 'SimpleCNN Mean 40.00 30.00 32.00 -10.00 -8.00' }),
  ).toBeInTheDocument()
})

it("drops a corruption's columns and chart when it is unchecked", async () => {
  const user = renderResultsPage()

  await user.click(await screen.findByRole('checkbox', { name: 'Noise' }))

  expect(screen.queryByRole('columnheader', { name: 'Noise Δ' })).not.toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'Noise Δ by training seed' })).not.toBeInTheDocument()
  expect(screen.getByRole('columnheader', { name: 'Blur Δ' })).toBeInTheDocument()
})

it('says so when the filters leave nothing to show', async () => {
  const user = renderResultsPage()

  await user.click(await screen.findByRole('checkbox', { name: '42' }))
  await user.click(screen.getByRole('checkbox', { name: '123' }))

  expect(screen.getByText('No runs match these filters.')).toBeInTheDocument()
  expect(screen.queryByRole('table')).not.toBeInTheDocument()
})

it('explains how to fill an empty database', async () => {
  renderResultsPage(() => Response.json([]))

  expect(await screen.findByText(/python -m src\.backfill/)).toBeInTheDocument()
})

it('shows the HTTP status when the API fails', async () => {
  renderResultsPage(() => new Response('Internal Server Error', { status: 500 }))

  expect(await screen.findByRole('alert')).toHaveTextContent('GET /results failed with HTTP 500')
})
