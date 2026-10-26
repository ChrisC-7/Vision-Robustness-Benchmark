import { api, ApiError } from './client'
import type { components } from './schema'

export type Prediction = components['schemas']['PredictionResponse']

/** Runs every loaded model on one image (POST /predict/compare). */
export async function comparePredictions(image: Blob): Promise<Prediction[]> {
  const { data, response } = await api.POST('/predict/compare', {
    // OpenAPI describes an uploaded file as a binary string, so that is the
    // generated type; bodySerializer is what actually builds the request body.
    body: { file: image as unknown as string },
    bodySerializer: ({ file }) => {
      const form = new FormData()
      form.append('file', file as unknown as Blob, 'input.png')
      return form
    },
  })
  if (data === undefined) throw new ApiError('POST /predict/compare', response.status)
  return data
}
