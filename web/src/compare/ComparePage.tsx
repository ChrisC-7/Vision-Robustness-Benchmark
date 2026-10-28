import { keepPreviousData, skipToken, useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'
import { comparePredictions, type Prediction } from '../api/compare'
import { Loading, QueryError } from '../components/QueryStates'
import { SERIES_COLORS } from '../results/chartStyle'
import { MODELS, MODEL_LABELS, type Condition, type Corruption } from '../results/evaluations'
import type { Pixels } from './corruptions'
import { loadModelInput, toImageData, toPngBlob } from './image'
import { applyCorruption, SLIDERS } from './settings'
import { useDebouncedValue } from './useDebouncedValue'

const CONDITION_CHOICES: Record<Condition, string> = {
  clean: 'None',
  blur: 'Gaussian blur',
  noise: 'Gaussian noise',
}

type UploadedImage = { id: number; name: string; pixels: Pixels }

export function ComparePage() {
  const [image, setImage] = useState<UploadedImage>()
  const [loadError, setLoadError] = useState<string>()
  const [condition, setCondition] = useState<Condition>('clean')
  const [levels, setLevels] = useState<Record<Corruption, number>>({
    blur: SLIDERS.blur.benchmark,
    noise: SLIDERS.noise.benchmark,
  })
  const uploads = useRef(0)

  // The preview follows the slider live; requests wait until it settles.
  // Debouncing both levels together (not the current one) means switching
  // corruption never pairs the new corruption with the old one's level.
  const settledLevels = useDebouncedValue(levels, 200)
  const level = condition === 'clean' ? 0 : levels[condition]
  const settledLevel = condition === 'clean' ? 0 : settledLevels[condition]

  const preview = useMemo(
    () => image && applyCorruption(image.pixels, condition, level),
    [image, condition, level],
  )

  // A POST, but a read: it has no side effects, so it is a query. That gives
  // caching (sliding back to an earlier value is instant) and keeps the last
  // predictions on screen, dimmed, while the next ones load.
  const predictions = useQuery({
    queryKey: ['compare', image?.id, condition, settledLevel],
    queryFn: image
      ? async () => comparePredictions(await toPngBlob(applyCorruption(image.pixels, condition, settledLevel)))
      : skipToken,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })

  async function handleFile(file: File | undefined) {
    if (file === undefined) return
    try {
      const pixels = await loadModelInput(file)
      uploads.current += 1
      setImage({ id: uploads.current, name: file.name, pixels })
      setLoadError(undefined)
    } catch {
      setLoadError(`Couldn't read ${file.name} as an image.`)
    }
  }

  return (
    <div className="compare-layout">
      <div className="compare-controls">
        <p className="section-note">
          Upload any image. It is squashed to 32×32, as the models were trained on, then corrupted the same way the
          benchmark does before both models classify it.
        </p>

        <label className="file-input">
          Image
          <input type="file" accept="image/*" onChange={(event) => void handleFile(event.target.files?.[0])} />
        </label>
        {loadError !== undefined && (
          <p className="status status-error" role="alert">
            {loadError}
          </p>
        )}

        <fieldset className="filter-group">
          <legend>Corruption</legend>
          {(Object.keys(CONDITION_CHOICES) as Condition[]).map((choice) => (
            <label key={choice}>
              <input
                type="radio"
                name="corruption"
                value={choice}
                checked={condition === choice}
                onChange={() => setCondition(choice)}
              />
              {CONDITION_CHOICES[choice]}
            </label>
          ))}
        </fieldset>

        {condition !== 'clean' && (
          <LevelSlider
            corruption={condition}
            value={levels[condition]}
            onChange={(value) => setLevels({ ...levels, [condition]: value })}
          />
        )}

        {image !== undefined && preview !== undefined && (
          <div className="previews">
            <PixelPreview pixels={image.pixels} caption="Model input, clean" />
            {condition !== 'clean' && (
              <PixelPreview pixels={preview} caption={`Model input, ${CONDITION_CHOICES[condition].toLowerCase()}`} />
            )}
          </div>
        )}
      </div>

      <div className="compare-results">
        {image === undefined ? (
          <p className="status">Choose an image to see both models' predictions.</p>
        ) : predictions.isPending ? (
          <Loading>Classifying…</Loading>
        ) : predictions.isError ? (
          <QueryError onRetry={() => void predictions.refetch()} retrying={predictions.isFetching}>
            Couldn't get predictions: {predictions.error.message}
          </QueryError>
        ) : (
          <PredictionList predictions={predictions.data} stale={predictions.isPlaceholderData} />
        )}
      </div>
    </div>
  )
}

function LevelSlider({
  corruption,
  value,
  onChange,
}: {
  corruption: Corruption
  value: number
  onChange: (value: number) => void
}) {
  const slider = SLIDERS[corruption]
  const inputId = `${corruption}-level`
  const listId = `${corruption}-benchmark`
  // Explicit for/id pairs: <output> is itself labelable, so a <label>
  // wrapping both it and the slider would name the output, not the slider.
  return (
    <div className="level-slider">
      <div className="level-slider-label">
        <label htmlFor={inputId}>{slider.label}</label>
        <output htmlFor={inputId}>{value.toFixed(slider.digits)}</output>
      </div>
      <input
        id={inputId}
        type="range"
        min={slider.min}
        max={slider.max}
        step={slider.step}
        value={value}
        list={listId}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <datalist id={listId}>
        <option value={slider.benchmark} />
      </datalist>
      <p className="hint">
        {slider.hint} The benchmark uses {slider.benchmark.toFixed(slider.digits)}.{' '}
        {value !== slider.benchmark && (
          <button type="button" className="link-button" onClick={() => onChange(slider.benchmark)}>
            Reset to benchmark
          </button>
        )}
      </p>
    </div>
  )
}

/** The 32×32 input drawn at its true size and scaled up with hard pixel edges. */
function PixelPreview({ pixels, caption }: { pixels: Pixels; caption: string }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    canvas.current?.getContext('2d')?.putImageData(toImageData(pixels), 0, 0)
  }, [pixels])
  return (
    <figure className="pixel-preview">
      <canvas ref={canvas} width={pixels.width} height={pixels.height} role="img" aria-label={caption} />
      <figcaption>{caption}</figcaption>
    </figure>
  )
}

function PredictionList({ predictions, stale }: { predictions: readonly Prediction[]; stale: boolean }) {
  const byModel = MODELS.flatMap((model) => predictions.filter((prediction) => prediction.model === model))
  return (
    <ul className={stale ? 'prediction-list is-stale' : 'prediction-list'} aria-label="Predictions" aria-busy={stale}>
      {byModel.map((prediction) => (
        <li key={prediction.model} className="prediction-card">
          <p className="prediction-model">
            <span className="legend-swatch" style={{ background: SERIES_COLORS[prediction.model] }} aria-hidden="true" />
            {MODEL_LABELS[prediction.model]}
          </p>
          <p className="prediction-class">{prediction.predicted_class}</p>
          <p className="prediction-confidence">{(prediction.confidence * 100).toFixed(1)}% confident</p>
          <div className="meter" aria-hidden="true">
            <div
              className="meter-fill"
              style={{ width: `${prediction.confidence * 100}%`, background: SERIES_COLORS[prediction.model] }}
            />
          </div>
        </li>
      ))}
    </ul>
  )
}
