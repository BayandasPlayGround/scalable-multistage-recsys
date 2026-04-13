from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from amazon_recsys.config.container import build_container
from amazon_recsys.config.settings import AppSettings
from amazon_recsys.web.router import router as web_router

from .routers.health import router as health_router
from .routers.models import router as models_router
from .routers.recommendations import router as recommendations_router


STATIC_DIR = Path(__file__).resolve().parents[1] / "web" / "static"


def create_app(settings: AppSettings | None = None) -> FastAPI:
    container = build_container(settings)
    app = FastAPI(
        title=container.settings.app_name,
        version=container.settings.app_version,
        debug=container.settings.debug,
    )
    app.state.container = container
    app.include_router(health_router)
    app.include_router(models_router)
    app.include_router(recommendations_router)
    app.include_router(web_router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()
