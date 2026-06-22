"""
Streamlit dashboard for real-time forecast monitoring.

Renders live time-series charts (actual vs predicted for Load and Price),
error distribution histograms, latency gauges, metric cards, and a
progress bar. Consumes data via WebSocket from the FastAPI server.

Reference: SYSTEM_DESIGN.md Sections 15.1, 15.2, 15.3
"""

import json
import time
from datetime import datetime
from threading import Thread

import plotly.graph_objects as go
import streamlit as st
import websockets.sync.client as ws_client

from config.settings import Settings


def main() -> None:
    """Streamlit dashboard entry point."""
    settings = Settings()

    st.set_page_config(
        page_title="RT Electricity Forecast",
        page_icon="⚡",
        layout="wide",
    )

    st.title("⚡ RT-ELECTRICITY-FORECAST")

    # ── Header status bar ────────────────────────────────────────────
    header_cols = st.columns(3)
    status_placeholder = header_cols[0].empty()
    logical_time_placeholder = header_cols[1].empty()
    progress_placeholder = header_cols[2].empty()

    # ── Charts ───────────────────────────────────────────────────────
    st.subheader("Load Forecast (MW)")
    load_chart_placeholder = st.empty()

    st.subheader("Price Forecast ($/MWh)")
    price_chart_placeholder = st.empty()

    # ── Metrics row ──────────────────────────────────────────────────
    st.subheader("Evaluation Metrics")
    metric_cols = st.columns(6)
    load_mae_ph = metric_cols[0].empty()
    load_rmse_ph = metric_cols[1].empty()
    load_mape_ph = metric_cols[2].empty()
    price_mae_ph = metric_cols[3].empty()
    price_rmse_ph = metric_cols[4].empty()
    price_mape_ph = metric_cols[5].empty()

    # ── Bottom row: latency + error distribution ─────────────────────
    bottom_cols = st.columns(2)
    latency_placeholder = bottom_cols[0].empty()
    error_hist_placeholder = bottom_cols[1].empty()

    # ── Progress bar ─────────────────────────────────────────────────
    progress_bar = st.progress(0)

    # ── Data stores ──────────────────────────────────────────────────
    timestamps: list[str] = []
    load_actuals: list[float] = []
    load_forecasts: list[float] = []
    price_actuals: list[float] = []
    price_forecasts: list[float] = []
    load_errors: list[float] = []

    # Metrics state
    current_metrics: dict = {
        "load_mae": 0.0, "load_rmse": 0.0, "load_mape": 0.0,
        "price_mae": 0.0, "price_rmse": 0.0, "price_mape": 0.0,
        "latency_p99_ms": 0.0,
    }
    events_count = 0
    total_events = 1  # avoid div-by-zero

    # ── WebSocket connection ─────────────────────────────────────────
    ws_url = f"ws://localhost:{settings.api_port}/ws/stream"
    max_display = 200  # show last 200 points for performance

    status_placeholder.markdown("**Status:** 🔴 Connecting...")

    try:
        with ws_client.connect(ws_url) as websocket:
            status_placeholder.markdown("**Status:** 🟢 Streaming")

            while True:
                try:
                    raw = websocket.recv(timeout=5.0)
                except TimeoutError:
                    continue

                msg = json.loads(raw)

                if msg.get("type") == "forecast":
                    events_count += 1
                    timestamps.append(msg["logical_timestamp"])
                    load_actuals.append(msg["load_actual_mw"])
                    load_forecasts.append(msg["load_forecast_mw"])
                    price_actuals.append(msg["price_actual_usd_mwh"])
                    price_forecasts.append(msg["price_forecast_usd_mwh"])
                    load_errors.append(msg["load_error_mw"])

                    # Trim to max_display
                    if len(timestamps) > max_display:
                        timestamps = timestamps[-max_display:]
                        load_actuals = load_actuals[-max_display:]
                        load_forecasts = load_forecasts[-max_display:]
                        price_actuals = price_actuals[-max_display:]
                        price_forecasts = price_forecasts[-max_display:]
                        load_errors = load_errors[-max_display:]

                    logical_time_placeholder.markdown(
                        f"**Logical Time:** {msg['logical_timestamp']}"
                    )

                    # Update charts every 10 events
                    if events_count % 10 == 0:
                        # Load chart
                        load_fig = go.Figure()
                        load_fig.add_trace(go.Scatter(
                            x=timestamps, y=load_actuals,
                            name="Actual", line=dict(color="royalblue"),
                        ))
                        load_fig.add_trace(go.Scatter(
                            x=timestamps, y=load_forecasts,
                            name="Predicted", line=dict(color="orange", dash="dash"),
                        ))
                        load_fig.update_layout(
                            height=300, margin=dict(l=0, r=0, t=30, b=0),
                            yaxis_title="Load (MW)",
                        )
                        load_chart_placeholder.plotly_chart(
                            load_fig, use_container_width=True
                        )

                        # Price chart
                        price_fig = go.Figure()
                        price_fig.add_trace(go.Scatter(
                            x=timestamps, y=price_actuals,
                            name="Actual", line=dict(color="green"),
                        ))
                        price_fig.add_trace(go.Scatter(
                            x=timestamps, y=price_forecasts,
                            name="Predicted", line=dict(color="red", dash="dash"),
                        ))
                        price_fig.update_layout(
                            height=300, margin=dict(l=0, r=0, t=30, b=0),
                            yaxis_title="Price ($/MWh)",
                        )
                        price_chart_placeholder.plotly_chart(
                            price_fig, use_container_width=True
                        )

                        # Error histogram
                        if len(load_errors) > 10:
                            hist_fig = go.Figure()
                            hist_fig.add_trace(go.Histogram(
                                x=load_errors, nbinsx=30,
                                marker_color="gray", opacity=0.7,
                            ))
                            hist_fig.update_layout(
                                title="Load Error Distribution",
                                height=250,
                                margin=dict(l=0, r=0, t=40, b=0),
                                xaxis_title="Error (MW)",
                            )
                            error_hist_placeholder.plotly_chart(
                                hist_fig, use_container_width=True
                            )

                elif msg.get("type") == "metrics_update":
                    current_metrics["load_mae"] = msg.get("load_mae", 0.0)
                    current_metrics["load_rmse"] = msg.get("load_rmse", 0.0)
                    current_metrics["price_mae"] = msg.get("price_mae", 0.0)
                    current_metrics["price_rmse"] = msg.get("price_rmse", 0.0)
                    current_metrics["latency_p99_ms"] = msg.get("latency_p99_ms", 0.0)

                    load_mae_ph.metric("Load MAE", f"{current_metrics['load_mae']:.1f} MW")
                    load_rmse_ph.metric("Load RMSE", f"{current_metrics['load_rmse']:.1f} MW")
                    load_mape_ph.metric("Load MAPE", f"{current_metrics.get('load_mape', 0):.2f}%")
                    price_mae_ph.metric("Price MAE", f"${current_metrics['price_mae']:.2f}/MWh")
                    price_rmse_ph.metric("Price RMSE", f"${current_metrics['price_rmse']:.2f}/MWh")
                    price_mape_ph.metric("Price MAPE", f"{current_metrics.get('price_mape', 0):.2f}%")

                    # Latency gauge
                    lat_fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=current_metrics["latency_p99_ms"],
                        title={"text": "p99 Latency (ms)"},
                        gauge={
                            "axis": {"range": [0, 5]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 1], "color": "lightgreen"},
                                {"range": [1, 2], "color": "lightyellow"},
                                {"range": [2, 5], "color": "lightcoral"},
                            ],
                        },
                    ))
                    lat_fig.update_layout(
                        height=250, margin=dict(l=0, r=0, t=0, b=0)
                    )
                    latency_placeholder.plotly_chart(
                        lat_fig, use_container_width=True
                    )

                elif msg.get("type") == "ping":
                    continue

                # Update progress
                if total_events > 1:
                    progress_pct = min(events_count / total_events, 1.0)
                    progress_bar.progress(progress_pct)
                    progress_placeholder.markdown(
                        f"**Progress:** {events_count}/{total_events} "
                        f"({progress_pct * 100:.1f}%)"
                    )

    except Exception as e:
        status_placeholder.markdown(f"**Status:** 🔴 Disconnected ({e})")
        st.error(
            f"Could not connect to WebSocket at {ws_url}. "
            f"Ensure the streaming server is running (make stream). Error: {e}"
        )


if __name__ == "__main__":
    main()
