import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, devices } from '@playwright/test'

const repoRoot = resolve(import.meta.dirname, '..')
// The repository's virtualenv when there is one (local runs), otherwise
// whatever python is on PATH (CI).
const venvPython = resolve(repoRoot, '.venv/bin/python')
const python = existsSync(venvPython) ? venvPython : 'python'

/**
 * End-to-end tests run the real stack: uvicorn serving the trained
 * checkpoints, and the Vite dev server. Locally, servers that are already
 * running on those ports are reused.
 */
export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: 'http://localhost:5173',
    trace: 'on-first-retry',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: [
    {
      command: `${python} -m uvicorn src.api:app --port 8000`,
      cwd: repoRoot,
      url: 'http://127.0.0.1:8000/health',
      reuseExistingServer: !process.env.CI,
      env: {
        VISION_SIMPLE_RUN_DIR: process.env.VISION_SIMPLE_RUN_DIR ?? 'runs/multiseed/experiment1/simple_s42',
        VISION_STRONGER_RUN_DIR: process.env.VISION_STRONGER_RUN_DIR ?? 'runs/multiseed/experiment1/stronger_s42',
      },
    },
    {
      command: 'npm run dev -- --port 5173 --strictPort',
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
    },
  ],
})
