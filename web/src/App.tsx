import { HealthStatus } from './components/HealthStatus'

function App() {
  return (
    <main className="app">
      <header>
        <h1>Robustness Explorer</h1>
        <p className="subtitle">
          How two small CIFAR-10 CNNs hold up under Gaussian blur and noise.
        </p>
      </header>
      <section aria-labelledby="backend-heading">
        <h2 id="backend-heading">Backend</h2>
        <HealthStatus />
      </section>
    </main>
  )
}

export default App
