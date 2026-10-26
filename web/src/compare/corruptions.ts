/**
 * The benchmark's corruptions (src/data.py), reimplemented for the browser.
 *
 * They run on the 32×32 image the model actually sees, so a slider's σ or
 * std means exactly what it means in the benchmark. Pixel values are worked
 * on in [0, 1], as torchvision's ToTensor produces, and only rounded back to
 * bytes at the end.
 */

/** RGBA bytes in row-major order: the same shape as the browser's ImageData. */
export type Pixels = {
  width: number
  height: number
  data: Uint8ClampedArray<ArrayBuffer>
}

/** The kernel size src/data.py's gaussian_blur_transform defaults to. */
export const BLUR_KERNEL_SIZE = 5

/** Matches torchvision's _get_gaussian_kernel1d: a sampled, normalized Gaussian. */
export function gaussianKernel(size: number, sigma: number): number[] {
  const half = (size - 1) / 2
  const weights = Array.from({ length: size }, (_, i) => Math.exp(-0.5 * ((i - half) / sigma) ** 2))
  const total = weights.reduce((sum, w) => sum + w, 0)
  return weights.map((w) => w / total)
}

/**
 * transforms.GaussianBlur(kernel_size, sigma): a separable Gaussian with
 * "reflect" padding at the edges, as torchvision pads. Alpha is left as is.
 */
export function gaussianBlur(image: Pixels, sigma: number, kernelSize = BLUR_KERNEL_SIZE): Pixels {
  const { width, height } = image
  const kernel = gaussianKernel(kernelSize, sigma)
  const half = (kernelSize - 1) / 2
  const input = toUnitFloats(image)
  const horizontal = new Float64Array(input.length)
  const output = new Float64Array(input.length)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0
        for (let k = 0; k < kernelSize; k++) {
          sum += kernel[k]! * input[(y * width + reflect(x + k - half, width)) * 4 + c]!
        }
        horizontal[(y * width + x) * 4 + c] = sum
      }
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        let sum = 0
        for (let k = 0; k < kernelSize; k++) {
          sum += kernel[k]! * horizontal[(reflect(y + k - half, height) * width + x) * 4 + c]!
        }
        output[(y * width + x) * 4 + c] = sum
      }
    }
  }
  return fromUnitFloats(output, image)
}

/**
 * GaussianNoise: adds N(mean, std²) to every channel of every pixel, then
 * clamps to [0, 1]. Seeded, so the same image and std always give the same
 * noise; the pattern differs from torch's RNG, but its distribution is the same.
 */
export function gaussianNoise(image: Pixels, std: number, seed: number, mean = 0): Pixels {
  const random = mulberry32(seed)
  const values = toUnitFloats(image)
  for (let i = 0; i < values.length; i++) {
    if (i % 4 !== 3) values[i] = Math.min(1, Math.max(0, values[i]! + mean + std * standardNormal(random)))
  }
  return fromUnitFloats(values, image)
}

/** torch's "reflect" padding: index -1 maps to 1, and n maps to n - 2. */
function reflect(index: number, size: number): number {
  if (index < 0) return -index
  if (index >= size) return 2 * (size - 1) - index
  return index
}

function toUnitFloats(image: Pixels): Float64Array {
  return Float64Array.from(image.data, (byte) => byte / 255)
}

/** Back to bytes; alpha is copied from the source image untouched. */
function fromUnitFloats(values: Float64Array, source: Pixels): Pixels {
  const data = new Uint8ClampedArray(values.length)
  for (let i = 0; i < values.length; i++) {
    data[i] = i % 4 === 3 ? source.data[i]! : Math.round(values[i]! * 255)
  }
  return { width: source.width, height: source.height, data }
}

/** A small, fast, seedable PRNG returning floats in [0, 1). */
function mulberry32(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Box–Muller transform: two uniforms in, one standard normal out. */
function standardNormal(random: () => number): number {
  const u = 1 - random() // in (0, 1], so log(u) is finite
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random())
}
