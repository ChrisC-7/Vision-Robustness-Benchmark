import { act, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { Loading, SLOW_REQUEST_MS } from './QueryStates'

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
})

it('explains a slow request as the free server waking up', () => {
  render(<Loading>Loading results…</Loading>)

  expect(screen.getByRole('status')).toHaveTextContent(/^Loading results…$/)

  act(() => {
    vi.advanceTimersByTime(SLOW_REQUEST_MS)
  })

  expect(screen.getByRole('status')).toHaveTextContent('waking it can take up to a minute')
})
