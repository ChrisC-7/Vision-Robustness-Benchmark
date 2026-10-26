import { expect, test, type Page } from '@playwright/test'

const IMAGE = 'e2e/fixtures/gradient.png'
const CIFAR_CLASS = /^(Airplane|Automobile|Bird|Cat|Deer|Dog|Frog|Horse|Ship|Truck)$/

async function uploadImage(page: Page) {
  await page.goto('/compare')
  await page.getByLabel('Image').setInputFiles(IMAGE)
}

/** Width and height from a PNG's IHDR chunk, which always starts at byte 16. */
function pngSize(body: Buffer): [number, number] {
  const start = body.indexOf(Buffer.from([0x89, 0x50, 0x4e, 0x47]))
  return [body.readUInt32BE(start + 16), body.readUInt32BE(start + 20)]
}

test('an uploaded image gets a prediction from each model', async ({ page }) => {
  await uploadImage(page)

  const cards = page.getByRole('list', { name: 'Predictions' }).getByRole('listitem')
  await expect(cards).toHaveCount(2)
  for (const [index, model] of ['SimpleCNN', 'StrongerCNN'].entries()) {
    const card = cards.nth(index)
    await expect(card).toContainText(model)
    await expect(card.locator('.prediction-class')).toHaveText(CIFAR_CLASS)
    await expect(card.locator('.prediction-confidence')).toHaveText(/^\d{1,3}\.\d% confident$/)
  }
})

test('choosing a corruption sends one request, with the corrupted 32×32 image', async ({ page }) => {
  // Bodies are read in a route handler: page.waitForRequest() reports no
  // body for a multipart upload built from a Blob, but routing does.
  const uploads: Buffer[] = []
  await page.route('**/predict/compare', async (route) => {
    uploads.push(route.request().postDataBuffer()!)
    await route.continue()
  })
  const predictions = page.getByRole('list', { name: 'Predictions' })

  await uploadImage(page)
  await expect(predictions.getByRole('listitem')).toHaveCount(2)
  await page.getByRole('radio', { name: 'Gaussian noise' }).check()
  await expect(page.getByRole('slider', { name: /Noise std/ })).toHaveValue('0.1')
  await expect(page.getByRole('img', { name: 'Model input, gaussian noise' })).toBeVisible()
  await expect.poll(() => uploads.length).toBe(2)
  await expect(predictions).toHaveAttribute('aria-busy', 'false')

  const [clean, noisy] = uploads
  expect(pngSize(clean!)).toEqual([32, 32])
  expect(pngSize(noisy!)).toEqual([32, 32])
  expect(noisy!.equals(clean!)).toBe(false)
  // Still two: the debounced slider and the query cache sent nothing extra.
  expect(uploads).toHaveLength(2)
})
