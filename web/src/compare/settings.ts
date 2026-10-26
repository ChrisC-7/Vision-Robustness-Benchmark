import type { Condition, Corruption } from '../results/evaluations'
import { gaussianBlur, gaussianNoise, type Pixels } from './corruptions'

/** Fixed, so a given image and std always get the same noise and the same prediction. */
export const NOISE_SEED = 42

type Slider = {
  label: string
  min: number
  max: number
  step: number
  /** The value src/engine.py's EvalConfig evaluates with. */
  benchmark: number
  digits: number
  hint: string
}

export const SLIDERS: Record<Corruption, Slider> = {
  blur: {
    label: 'Blur σ',
    min: 0.1,
    max: 3,
    step: 0.1,
    benchmark: 1.0,
    digits: 1,
    hint: 'In pixels of the 32×32 image, with a 5×5 kernel.',
  },
  noise: {
    label: 'Noise std',
    min: 0.01,
    max: 0.5,
    step: 0.01,
    benchmark: 0.1,
    digits: 2,
    hint: 'On the 0–1 pixel scale.',
  },
}

export function applyCorruption(image: Pixels, condition: Condition, level: number): Pixels {
  switch (condition) {
    case 'clean':
      return image
    case 'blur':
      return gaussianBlur(image, level)
    case 'noise':
      return gaussianNoise(image, level, NOISE_SEED)
  }
}
