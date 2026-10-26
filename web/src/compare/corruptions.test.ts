import { describe, expect, it } from 'vitest'
import { gaussianBlur, gaussianKernel, gaussianNoise, type Pixels } from './corruptions'

function pixels(width: number, height: number, rgba: number[]): Pixels {
  return { width, height, data: new Uint8ClampedArray(rgba) }
}

// A random 6×4 RGB image, and torchvision's output for it: produced with
// transforms.GaussianBlur(kernel_size=5, sigma) on the image as floats in
// [0, 1], then rounded back to bytes.
const INPUT = pixels(6, 4, [
  172, 88, 142, 255, 47, 81, 167, 255, 117, 165, 32, 255, 192, 25, 193, 255, 67, 77, 9, 255, 251, 72, 185, 255,
  195, 9, 127, 255, 103, 148, 32, 255, 9, 115, 31, 255, 211, 208, 202, 255, 21, 243, 244, 255, 242, 197, 151, 255,
  36, 254, 163, 255, 87, 79, 254, 255, 70, 175, 203, 255, 216, 192, 114, 255, 88, 82, 183, 255, 140, 99, 28, 255,
  58, 216, 34, 255, 193, 177, 128, 255, 230, 243, 128, 255, 39, 29, 164, 255, 87, 147, 53, 255, 174, 147, 133, 255,
])
const TORCHVISION_BLUR: Record<number, number[]> = {
  1.0: [
    113, 98, 115, 255, 103, 110, 107, 255, 105, 129, 109, 255, 125, 140, 137, 255, 132, 146, 151, 255, 140, 147,
    151, 255, 109, 113, 123, 255, 102, 122, 118, 255, 106, 139, 121, 255, 124, 149, 143, 255, 128, 151, 152, 255,
    134, 150, 150, 255, 102, 149, 144, 255, 107, 149, 146, 255, 117, 153, 146, 255, 125, 149, 146, 255, 123, 141,
    139, 255, 125, 137, 130, 255, 103, 168, 147, 255, 114, 164, 153, 255, 127, 159, 154, 255, 125, 144, 143, 255,
    119, 131, 127, 255, 121, 127, 116, 255,
  ],
  2.5: [
    86, 122, 123, 255, 110, 126, 134, 255, 108, 136, 140, 255, 122, 142, 138, 255, 121, 148, 145, 255, 142, 150,
    157, 255, 103, 134, 111, 255, 118, 131, 124, 255, 112, 138, 130, 255, 124, 144, 133, 255, 120, 149, 139, 255,
    134, 146, 153, 255, 102, 147, 140, 255, 114, 142, 145, 255, 111, 144, 142, 255, 126, 137, 138, 255, 122, 137,
    132, 255, 132, 130, 137, 255, 102, 149, 135, 255, 115, 149, 140, 255, 109, 154, 145, 255, 123, 152, 142, 255,
    117, 155, 145, 255, 129, 152, 149, 255,
  ],
}

describe('gaussianKernel', () => {
  it('is symmetric and sums to 1', () => {
    const kernel = gaussianKernel(5, 1)

    expect(kernel.reduce((sum, w) => sum + w, 0)).toBeCloseTo(1, 12)
    expect(kernel[0]).toBeCloseTo(kernel[4]!, 12)
    expect(kernel[2]).toBeGreaterThan(kernel[1]!)
  })
})

describe('gaussianBlur', () => {
  it.each([1.0, 2.5])('matches torchvision GaussianBlur(5, sigma=%s) byte for byte', (sigma) => {
    expect(Array.from(gaussianBlur(INPUT, sigma).data)).toEqual(TORCHVISION_BLUR[sigma])
  })

  it('leaves a flat image unchanged', () => {
    const flat = pixels(3, 3, Array.from({ length: 9 }, () => [10, 120, 250, 255]).flat())

    expect(gaussianBlur(flat, 2).data).toEqual(flat.data)
  })
})

describe('gaussianNoise', () => {
  const gray = pixels(16, 16, Array.from({ length: 256 }, () => [128, 128, 128, 255]).flat())

  it('is the identity at std 0', () => {
    expect(gaussianNoise(INPUT, 0, 42).data).toEqual(INPUT.data)
  })

  it('gives the same noise for the same seed, and different noise for another', () => {
    expect(gaussianNoise(gray, 0.1, 42).data).toEqual(gaussianNoise(gray, 0.1, 42).data)
    expect(gaussianNoise(gray, 0.1, 42).data).not.toEqual(gaussianNoise(gray, 0.1, 7).data)
  })

  it('has roughly the requested spread, in [0, 1] units', () => {
    const noisy = gaussianNoise(gray, 0.1, 42).data
    const deviations = Array.from(noisy)
      .filter((_, i) => i % 4 !== 3)
      .map((byte) => byte / 255 - 128 / 255)
    const sd = Math.sqrt(deviations.reduce((sum, d) => sum + d * d, 0) / deviations.length)

    expect(sd).toBeGreaterThan(0.09)
    expect(sd).toBeLessThan(0.11)
  })

  it('clamps out-of-range values instead of wrapping, and never touches alpha', () => {
    const channels = (image: Pixels) => Array.from(image.data).filter((_, i) => i % 4 !== 3)

    // A mean shift of ±2 pushes every channel far past the [0, 1] range.
    expect(channels(gaussianNoise(INPUT, 0.1, 42, 2)).every((byte) => byte === 255)).toBe(true)
    expect(channels(gaussianNoise(INPUT, 0.1, 42, -2)).every((byte) => byte === 0)).toBe(true)
    expect(Array.from(gaussianNoise(INPUT, 0.1, 42, -2).data).filter((_, i) => i % 4 === 3)).toEqual(
      Array(24).fill(255),
    )
  })
})
