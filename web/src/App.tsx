import { HealthStatus } from './components/HealthStatus'
import { ResultsPage } from './results/ResultsPage'

function App() {
  return (
    <main className="app">
      <header>
        <h1>Robustness Explorer</h1>
        <p className="subtitle">
          How two small CIFAR-10 CNNs hold up under Gaussian blur and noise.
        </p>
      </header>
      <section aria-labelledby="results-heading">
        <h2 id="results-heading">Results</h2>
        <ResultsPage />
      </section>
      <section aria-labelledby="backend-heading">
        <h2 id="backend-heading">Backend</h2>
        <HealthStatus />
      </section>
    </main>
  )
}

export default App
