"""Contract test between the API and the web frontend's generated types."""

from src.export_openapi import OPENAPI_PATH, render_openapi


def test_committed_openapi_schema_is_up_to_date():
    assert OPENAPI_PATH.read_text() == render_openapi(), (
        "web/openapi.json no longer matches src/api.py. Run "
        "`python -m src.export_openapi`, then `npm run gen:api` in web/."
    )
