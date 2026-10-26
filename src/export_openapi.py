"""Write the API's OpenAPI schema to ``web/openapi.json``.

The web frontend generates its TypeScript types from that file, so after
changing any request or response model in :mod:`api`, regenerate both:

    python -m src.export_openapi
    cd web && npm run gen:api

``tests/test_openapi.py`` fails while the committed file is out of date, so
a backend change cannot silently drift away from the frontend's types.
Building the schema never runs the app's lifespan, so no checkpoints or env
vars are needed.
"""

import json
from pathlib import Path

from .api import app

OPENAPI_PATH = Path(__file__).resolve().parent.parent / "web" / "openapi.json"


def render_openapi() -> str:
    """Returns the schema exactly as it is written to ``OPENAPI_PATH``."""
    return json.dumps(app.openapi(), indent=2) + "\n"


if __name__ == "__main__":
    OPENAPI_PATH.parent.mkdir(exist_ok=True)
    OPENAPI_PATH.write_text(render_openapi())
    print(f"Wrote {OPENAPI_PATH}")
