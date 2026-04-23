"""FastAPI entrypoint for the private image description service."""

from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from service.app.core import storage
from service.app.core.config import settings
from service.app.routers import description


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    Path(settings.tmp_dir).mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="Private Description Service",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(description.router)
