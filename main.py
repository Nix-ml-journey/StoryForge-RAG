from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

# Ensure `src/` is on sys.path so `import storyforge...` works when running `py main.py`
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from storyforge.api.create_eval_routes import create_eval_router  # noqa: E402
from storyforge.api.data_routers import book_router  # noqa: E402
from storyforge.api.orchestration_routes import orchestration_router  # noqa: E402
from storyforge.api.vector_store_check import (  # noqa: E402
    vector_store_check_router,
    vector_store_router,
)
from storyforge.config.config import load_config  # noqa: E402


def create_app() -> FastAPI:
    cfg = load_config()
    app = FastAPI(
        title="StoryForge API",
        version="0.1.0",
    )

    # Core routers
    app.include_router(book_router)
    app.include_router(orchestration_router)
    app.include_router(vector_store_router)
    app.include_router(vector_store_check_router)
    app.include_router(create_eval_router)

    # Simple landing route
    @app.get("/", tags=["Status"])
    def root():
        return {
            "ok": True,
            "message": "StoryForge is running",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "port": int(cfg.get("Port") or 8000),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = load_config()
    port = int(cfg.get("Port") or 8000)
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=False)

