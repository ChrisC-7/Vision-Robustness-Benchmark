import type { Pixels } from './corruptions'

export const MODEL_INPUT_SIZE = 32

/**
 * Decodes an image file and squashes it to 32×32, as the API's
 * transforms.Resize((32, 32)) does: no crop, aspect ratio not kept. Sending
 * this 32×32 image makes the API's own resize a no-op, so the corruptions
 * applied here are exactly what the model sees.
 *
 * The browser's downscaling filter is not PIL's bilinear filter, so a clean
 * upload can differ from what /predict computes by a few levels per pixel.
 */
export async function loadModelInput(file: Blob): Promise<Pixels> {
  const bitmap = await createImageBitmap(file)
  try {
    const context = context2d(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    context.imageSmoothingQuality = 'high'
    context.drawImage(bitmap, 0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    const { width, height, data } = context.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    return { width, height, data }
  } finally {
    bitmap.close()
  }
}

export function toPngBlob(image: Pixels): Promise<Blob> {
  const context = context2d(image.width, image.height)
  context.putImageData(toImageData(image), 0, 0)
  return context.canvas.convertToBlob({ type: 'image/png' })
}

export function toImageData(image: Pixels): ImageData {
  return new ImageData(image.data, image.width, image.height)
}

function context2d(width: number, height: number): OffscreenCanvasRenderingContext2D {
  const context = new OffscreenCanvas(width, height).getContext('2d')
  if (context === null) throw new Error('This browser cannot draw to a canvas.')
  return context
}
