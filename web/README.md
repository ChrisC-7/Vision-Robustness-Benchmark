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

Open http://localhost:5173. The app calls the API at `VITE_API_BASE_URL` (default `http://127.0.0.1:8000`; see `.env.example`). The API only accepts browser requests from the origins in its `VISION_CORS_ORIGINS`, which defaults to `http://localhost:5173`, so open the app on `localhost`, not `127.0.0.1`.

## Scripts

| Script | What it does |
| --- | --- |
| `npm run dev` | Vite dev server with hot reload |
| `npm run build` | type-check, then build to `dist/` |
| `npm run typecheck` | `tsc -b` only |
| `npm run lint` | Oxlint, including type-aware rules such as `no-floating-promises` |
| `npm test` | Vitest + Testing Library in jsdom |
| `npm run gen:api` | regenerate `src/api/schema.d.ts` from `openapi.json` |

## How the API types stay in sync

```text
src/api.py (Pydantic models)
  └─ python -m src.export_openapi ──> web/openapi.json
       └─ npm run gen:api ──> web/src/api/schema.d.ts
            └─ src/api/client.ts (openapi-fetch, typed by schema.d.ts)
```

Both generated files are committed. After changing a request or response model in `src/api.py`, run both commands. `tests/test_openapi.py` fails while `openapi.json` is stale, and once the types are regenerated, `npm run typecheck` fails anywhere the frontend still uses the old shape.

`openapi-typescript` 7.x declares a TypeScript 5 peer dependency, but this project uses TypeScript 6. The `overrides` entry in `package.json` lets it run on TypeScript 6; the generated types are type-checked with every build.
