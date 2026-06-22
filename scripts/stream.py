"""
CLI entrypoint for streaming simulation.

Full lifecycle:
1. Load configuration from .env / config/settings.py
2. Load serialized model from artifacts/model.lgb
3. Initialize FeatureBuffer, PricingEngine, MetricsCollector
4. Initialize TokenBucketRateLimiter
5. Initialize LogicalTimeReplayEngine with test split data
6. Launch FastAPI server (uvicorn, port 8000)
7. Start asyncio event loop with streaming pipeline
8. On completion: flush metrics, generate evaluation report

Reference: SYSTEM_DESIGN.md Section 17.3
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

import uvicorn

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings
from core.data_loader import DataLoader
from streaming.pipeline import StreamingPipeline
from streaming.rate_limiter import TokenBucketRateLimiter
from streaming.replay_engine import LogicalTimeReplayEngine
from api.app import create_app, set_pipeline
from api.websocket import manager


async def run_streaming(settings: Settings) -> None:
    """Run the full streaming simulation lifecycle."""
    logger = logging.getLogger(__name__)

    # Step 1-2: Load data and split
    logger.info("Loading dataset and splitting...")
    loader = DataLoader(settings)
    df = loader.load()
    _, stream_df = loader.split(df)
    logger.info("Streaming replay buffer: %d events.", len(stream_df))

    # Step 3: Initialize pipeline components
    pipeline = StreamingPipeline(settings)

    # Step 4: Initialize rate limiter
    rate_limiter = TokenBucketRateLimiter(settings=settings)

    # Step 5: Initialize replay engine
    replay_engine = LogicalTimeReplayEngine(
        stream_df=stream_df,
        rate_limiter=rate_limiter,
        settings=settings,
    )

    # Set pipeline globally for API access
    set_pipeline(pipeline)

    # Set WebSocket broadcast callback
    pipeline.set_ws_broadcast(manager.broadcast)

    # Step 6: Create and launch FastAPI app
    app = create_app(settings)

    config = uvicorn.Config(
        app=app,
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Step 7-8: Run API server and streaming pipeline concurrently
    async def run_api() -> None:
        await server.serve()

    async def run_pipeline() -> dict:
        # Small delay to let API server start
        await asyncio.sleep(1.0)

        logger.info("Starting streaming pipeline...")
        final_metrics = await pipeline.run(replay_engine)

        # Save evaluation report
        report_path = Path("artifacts/evaluation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(final_metrics, f, indent=2, default=str)
        logger.info("Evaluation report saved to %s", report_path)

        return final_metrics

    # Run both concurrently
    api_task = asyncio.create_task(run_api())
    pipeline_task = asyncio.create_task(run_pipeline())

    # Wait for pipeline to complete
    final_metrics = await pipeline_task

    logger.info("=" * 60)
    logger.info("STREAMING SIMULATION COMPLETE")
    logger.info("-" * 60)
    logger.info(
        "Load MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        final_metrics["load_metrics"]["mae"],
        final_metrics["load_metrics"]["rmse"],
        final_metrics["load_metrics"]["mape"],
    )
    logger.info(
        "Price MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
        final_metrics["price_metrics"]["mae"],
        final_metrics["price_metrics"]["rmse"],
        final_metrics["price_metrics"]["mape"],
    )
    logger.info("=" * 60)

    # Shutdown server gracefully
    server.should_exit = True
    await api_task


def main() -> None:
    """Main entry point for streaming simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("RT-ELECTRICITY-FORECAST: Streaming Simulation")
    logger.info("=" * 60)

    settings = Settings()
    asyncio.run(run_streaming(settings))


if __name__ == "__main__":
    main()
