"""
FastAPI application factory.

Creates and configures the FastAPI app with lifespan events, CORS,
and includes routes and WebSocket handlers.

Reference: SYSTEM_DESIGN.md Section 14, 17.3
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import Settings
from streaming.pipeline import StreamingPipeline

logger = logging.getLogger(__name__)

# Global pipeline instance shared across the app
_pipeline: StreamingPipeline | None = None


def get_pipeline() -> StreamingPipeline:
    """Get the global streaming pipeline instance."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return _pipeline


def set_pipeline(pipeline: StreamingPipeline) -> None:
    """Set the global streaming pipeline instance."""
    global _pipeline
    _pipeline = pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: startup and shutdown events."""
    logger.info("FastAPI application starting up.")
    yield
    logger.info("FastAPI application shutting down.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Configuration settings. Uses defaults if not provided.

    Returns:
        Configured FastAPI application instance.
    """
    s = settings or Settings()

    app = FastAPI(
        title="RT Electricity Forecast",
        description=(
            "Real-Time Electricity Price Forecasting via Two-Stage "
            "Load Prediction with Simulated Streaming Infrastructure"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    from api.routes import router as routes_router
    from api.websocket import router as ws_router

    app.include_router(routes_router)
    app.include_router(ws_router)

    return app
