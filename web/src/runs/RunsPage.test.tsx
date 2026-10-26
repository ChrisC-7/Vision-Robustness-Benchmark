import { screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import type { RunSummary } from '../api/runs'
import { renderPage } from '../test/render'
import { RunsPage } from './RunsPage'

const RUNS: RunSummary[] = [
  {
    run_dir: 'runs/multiseed/experiment1/simple_s42',
    run_name: 'simple_s42',
    model_name: 'simple',
    training_seed: 42,
    split_seed: 42,
    best_epoch: 29,
    best_val_accuracy: 40.7,
  },
  {
    run_dir: 'runs/legacy_run',
    run_name: 'legacy_run',
    model_name: 'stronger',
    training_seed: null,
    split_seed: null,
    best_epoch: null,
    best_val_accuracy: null,
  },
]

function renderRunsPage(respond = () => Response.json(RUNS)) {
  return renderPage([{ path: '/runs', element: <RunsPage /> }], '/runs', respond)
}

it('lists each run with its best epoch, linking to its detail page', async () => {
  renderRunsPage()

  expect(
    await screen.findByRole('row', { name: 'simple_s42 experiment1 SimpleCNN 42 42 29 40.70' }),
  ).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'simple_s42' })).toHaveAttribute(
    'href',
    '/runs/runs/multiseed/experiment1/simple_s42',
  )
})

it('shows unknown fields as such instead of guessing', async () => {
  renderRunsPage()

  expect(await screen.findByRole('row', { name: 'legacy_run runs StrongerCNN unknown unknown — —' })).toBeInTheDocument()
})

it('opens a run when its link is clicked', async () => {
  const { user } = renderRunsPage()

  await user.click(await screen.findByRole('link', { name: 'simple_s42' }))

  expect(screen.getByText('Navigated away')).toBeInTheDocument()
})

it('explains how to fill an empty database', async () => {
  renderRunsPage(() => Response.json([]))

  expect(await screen.findByText(/No training runs are recorded yet/)).toHaveTextContent('python -m src.backfill')
})
