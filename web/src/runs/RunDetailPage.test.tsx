import { screen } from '@testing-library/react'
import { expect, it } from 'vitest'
import { apiBaseUrl } from '../api/client'
import type { RunDetail } from '../api/runs'
import { renderPage } from '../test/render'
import { RunDetailPage } from './RunDetailPage'

const RUN: RunDetail = {
  run_dir: 'runs/multiseed/experiment1/simple_s42',
  run_name: 'simple_s42',
  model_name: 'simple',
  training_seed: 42,
  split_seed: 42,
  best_epoch: 29,
  best_val_accuracy: 40.7,
  train_config: { model_name: 'simple', learning_rate: 0.03, epochs: 30, augment: false },
}

// The app's own pair of routes: an exact /runs outranks the /runs/* splat,
// which would otherwise also match /runs with an empty splat.
function renderRunDetail(url: string, respond: (request: Request) => Response) {
  return renderPage(
    [
      { path: '/runs', element: <p>Runs list</p> },
      { path: '/runs/*', element: <RunDetailPage /> },
    ],
    url,
    respond,
  )
}

it('fetches the run named by the URL and shows its best epoch and config', async () => {
  const { fetchMock } = renderRunDetail('/runs/runs/multiseed/experiment1/simple_s42', () => Response.json(RUN))

  expect(await screen.findByRole('heading', { name: 'simple_s42' })).toBeInTheDocument()
  // The run_dir's slashes travel percent-encoded, as one path parameter.
  expect(fetchMock.mock.calls[0]?.[0].url).toBe(`${apiBaseUrl}/runs/runs%2Fmultiseed%2Fexperiment1%2Fsimple_s42`)
  expect(screen.getByText('Best epoch').nextElementSibling).toHaveTextContent('29 of 30')
  expect(screen.getByText('Best validation accuracy').nextElementSibling).toHaveTextContent('40.70%')
  expect(screen.getByText('learning_rate').closest('div')).toHaveTextContent('learning_rate0.03')
  expect(screen.getByText('augment').closest('div')).toHaveTextContent('augmentfalse')
})

it('says a run is not recorded on a 404, without retrying', async () => {
  const { fetchMock } = renderRunDetail('/runs/runs/nope', () => Response.json({ detail: 'nope' }, { status: 404 }))

  expect(await screen.findByText(/No run is recorded for/)).toHaveTextContent('runs/nope')
  expect(fetchMock).toHaveBeenCalledTimes(1)
})

it('retries a server error once, then offers a retry button', async () => {
  const { fetchMock } = renderRunDetail('/runs/runs/x', () => new Response('boom', { status: 500 }))

  expect(await screen.findByRole('alert')).toHaveTextContent('GET /runs/runs/x failed with HTTP 500')
  expect(fetchMock).toHaveBeenCalledTimes(2)
})

it('links back to the list', async () => {
  const { user } = renderRunDetail('/runs/runs/multiseed/experiment1/simple_s42', () => Response.json(RUN))

  await user.click(await screen.findByRole('link', { name: '← All runs' }))

  expect(screen.getByText('Runs list')).toBeInTheDocument()
})
