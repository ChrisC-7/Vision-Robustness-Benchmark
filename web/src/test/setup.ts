import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

// Vitest globals are off, so Testing Library can't register its own
// afterEach cleanup; without this, each test's DOM leaks into the next.
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})
