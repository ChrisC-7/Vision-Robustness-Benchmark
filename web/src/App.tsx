import { lazy, Suspense, type ReactNode } from 'react'
import { NavLink, Route, Routes } from 'react-router'
import { HealthStatus } from './components/HealthStatus'

// Each page is its own chunk, so Recharts (results only) never loads on the
// compare page, and neither page's code is in the initial bundle.
const ResultsPage = lazy(() => import('./results/ResultsPage').then((m) => ({ default: m.ResultsPage })))
const ComparePage = lazy(() => import('./compare/ComparePage').then((m) => ({ default: m.ComparePage })))
const RunsPage = lazy(() => import('./runs/RunsPage').then((m) => ({ default: m.RunsPage })))
const RunDetailPage = lazy(() => import('./runs/RunDetailPage').then((m) => ({ default: m.RunDetailPage })))

function App() {
  return (
    <main className="app">
      <header className="app-header">
        <div>
          <h1>Robustness Explorer</h1>
          <p className="subtitle">How two small CIFAR-10 CNNs hold up under Gaussian blur and noise.</p>
        </div>
        <nav aria-label="Pages">
          <NavLink to="/" end>
            Results
          </NavLink>
          <NavLink to="/compare">Compare</NavLink>
          <NavLink to="/runs">Runs</NavLink>
        </nav>
      </header>

      <Suspense fallback={<p role="status">Loading…</p>}>
        <Routes>
          <Route path="/" element={<Page title="Results"><ResultsPage /></Page>} />
          <Route path="/compare" element={<Page title="Compare models"><ComparePage /></Page>} />
          <Route path="/runs" element={<Page title="Training runs"><RunsPage /></Page>} />
          {/* A splat, not :runDir, because a run_dir contains slashes. */}
          <Route path="/runs/*" element={<Page title="Training run"><RunDetailPage /></Page>} />
          <Route path="*" element={<Page title="Page not found"><p>Nothing lives at this address.</p></Page>} />
        </Routes>
      </Suspense>

      <section aria-labelledby="backend-heading">
        <h2 id="backend-heading">Backend</h2>
        <HealthStatus />
      </section>
    </main>
  )
}

function Page({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section aria-labelledby="page-heading">
      <h2 id="page-heading">{title}</h2>
      {children}
    </section>
  )
}

export default App
