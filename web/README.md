# Robustness Explorer (web)

React + TypeScript frontend for the FastAPI server in [`../src/api.py`](../src/api.py). Built with Vite, TanStack Query for requests, and an API client whose types are generated from the backend's OpenAPI schema.

## Run it

Start the API first, from the repository root (see the main [README](../README.md#usage)):

```bash
uvicorn src.api:app --reload
```

Then, from `web/`:

```bash
npm install
npm run dev
```

Open http://localhost:5173. The results page charts every evaluation recorded in the API's tracking database (fill it with `python -m src.backfill`; see the main README), with filters by model, training seed and corruption, and a table of every charted value. The compare page (`/compare`) takes any image, squashes it to the 32×32 the models see, applies the benchmark's Gaussian blur or noise in the browser with a slider for its strength, and shows both models' predictions side by side. The app calls the API at `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`; see `.env.example`). The API only accepts browser requests from the origins in its `VISION_CORS_ORIGINS`, which defaults to `http://localhost:5173`, so open the app on `localhost`, not `127.0.0.1`.

## Scripts

| Script | What it does |
| --- | --- |
| `npm run dev` | Vite dev server with hot reload |
| `npm run build` | type-check, then build to `dist/` |
| `npm run typecheck` | `tsc -b` only |
| `npm run lint` | Oxlint, including type-aware rules such as `no-floating-promises` |
| `npm test` | Vitest + Testing Library in jsdom |
| `npm run test:e2e` | Playwright end-to-end tests against the real API and checkpoints |
| `npm run gen:api` | regenerate `src/api/schema.d.ts` from `openapi.json` |

## End-to-end tests

`npm run test:e2e` starts uvicorn on port 8000 (serving `runs/multiseed/experiment1/{simple,stronger}_s42` unless `VISION_SIMPLE_RUN_DIR` / `VISION_STRONGER_RUN_DIR` say otherwise) and the Vite dev server on 5173, or reuses ones already running there. It then uploads an image on the compare page and checks the following:

- each model returns a CIFAR-10 class with a confidence;
- choosing a corruption sends exactly one new request;
- that request carries a 32×32 PNG that differs from the clean one.

The first run needs a browser:

```bash
npx playwright install chromium
```

On Linux, Chromium also needs a few system libraries. `npx playwright install --with-deps chromium` installs them, or on Ubuntu run `sudo apt install libnspr4 libnss3 libasound2t64`.

## Corruptions in the browser

`src/compare/corruptions.ts` reimplements the benchmark's transforms from `src/data.py`. The compare page applies them to the 32×32 image the model sees, so σ and std mean exactly what they mean in the benchmark.

- **Blur** matches torchvision's `GaussianBlur(5, σ)` byte for byte. The unit tests check this against torchvision's own output at two values of σ.
- **Noise** has the same distribution as the benchmark's, but it comes from a different seeded RNG, so the noise pattern differs.
- **Resizing** uses the browser's downscaling filter, not PIL's bilinear filter. A large clean upload can therefore differ from what `POST /predict` sees by a few levels per pixel. Images that are already 32×32, like CIFAR-10, are not resized and match exactly.

## How the API types stay in sync

```text
src/api.py (Pydantic models)
  └─ python -m src.export_openapi ──> web/openapi.json
       └─ npm run gen:api ──> web/src/api/schema.d.ts
            └─ src/api/client.ts (openapi-fetch, typed by schema.d.ts)
```

Both generated files are committed. After changing a request or response model in `src/api.py`, run both commands. `tests/test_openapi.py` fails while `openapi.json` is stale, and once the types are regenerated, `npm run typecheck` fails anywhere the frontend still uses the old shape.

`openapi-typescript` 7.x declares a TypeScript 5 peer dependency, but this project uses TypeScript 6. The `overrides` entry in `package.json` lets it run on TypeScript 6; the generated types are type-checked with every build.
