"""Generate an interactive dashboard for a year's worth of lake sensor readings.

The script expects a CSV with columns: ``timestamp`` (ISO 8601), ``ph``,
``turbidity``, ``temperature``, and ``do_level``. Run it like:

    python visualize_lake_readings.py sample_lake_readings.csv --output lake_dashboard.html

The output HTML contains:
- Multi-metric time series with daily rolling averages and a range slider.
- Hour-of-day heatmaps to reveal diurnal patterns across the year.
- Monthly box plots to highlight seasonal trends.
- A correlation matrix for quick relationship checks between sensors.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

SENSOR_COLUMNS: Tuple[str, ...] = ("ph", "turbidity", "temperature", "do_level")


def load_lake_data(csv_path: Path) -> pd.DataFrame:
    """Load and enrich the lake readings CSV."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    missing = [col for col in SENSOR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {', '.join(missing)}")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month_name()

    # 24-hour rolling mean (8 readings per day) for smoother visual trends
    for col in SENSOR_COLUMNS:
        df[f"{col}_roll_day"] = df[col].rolling(window=8, min_periods=3).mean()

    return df


def build_timeseries(df: pd.DataFrame) -> go.Figure:
    """Create a stacked time series with raw and smoothed traces."""
    fig = make_subplots(
        rows=len(SENSOR_COLUMNS),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[col.replace("_", " ").title() for col in SENSOR_COLUMNS],
    )

    for idx, col in enumerate(SENSOR_COLUMNS, start=1):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[col],
                mode="lines",
                line=dict(color="#7f8fa6", width=1),
                name=f"{col.title()} (raw)",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>",
                showlegend=(idx == 1),
            ),
            row=idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df[f"{col}_roll_day"],
                mode="lines",
                line=dict(color="#273c75", width=2),
                name=f"{col.title()} (24h avg)",
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>",
                showlegend=(idx == 1),
            ),
            row=idx,
            col=1,
        )

    fig.update_layout(
        height=1200,
        title="Full-year sensor time series",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=40, l=70, r=40),
        xaxis4=dict(rangeslider=dict(visible=True), type="date"),
    )
    fig.update_yaxes(matches=None, showgrid=True, zeroline=False)
    return fig


def build_hourly_heatmap(df: pd.DataFrame, metric: str) -> go.Figure:
    """Heatmap showing hour-of-day vs. day-of-year for a metric."""
    day_of_year = df["timestamp"].dt.dayofyear
    matrix = pd.pivot_table(
        df.assign(day_of_year=day_of_year),
        values=metric,
        index="hour",
        columns="day_of_year",
        aggfunc="mean",
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="Viridis",
            colorbar_title=metric.title(),
            hovertemplate="Day %{x}, Hour %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=400,
        title=f"{metric.title()} diurnal pattern (hour vs. day of year)",
        xaxis_title="Day of year",
        yaxis_title="Hour of day",
    )
    return fig


def build_monthly_boxplots(df: pd.DataFrame) -> go.Figure:
    """Seasonal distribution overview."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[c.title() for c in SENSOR_COLUMNS])
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for col, (r, c) in zip(SENSOR_COLUMNS, positions):
        fig.add_trace(
            go.Box(
                x=df["month"],
                y=df[col],
                boxmean=True,
                name=col.title(),
                marker_color="#44bd32",
                hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=r,
            col=c,
        )
    fig.update_layout(
        height=800,
        title="Monthly distributions",
        margin=dict(t=60, b=40, l=40, r=40),
    )
    return fig


def build_correlation(df: pd.DataFrame) -> go.Figure:
    """Simple correlation matrix for the sensor signals."""
    corr = df[list(SENSOR_COLUMNS)].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Sensor correlation matrix",
    )
    fig.update_layout(height=500, margin=dict(t=60, b=40, l=40, r=40))
    return fig


def build_dashboard_html(figures: Iterable[go.Figure], title: str) -> str:
    """Serialize multiple Plotly figures into a single HTML document."""
    figure_html = "\n".join(
        pio.to_html(fig, include_plotlyjs=False, full_html=False, default_height="100%")
        for fig in figures
    )
    template = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1.0' />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; margin: 0; padding: 0 1rem 2rem; background: #f5f6fa; }}
    header {{ padding: 1.5rem 0 0.5rem; }}
    h1 {{ margin: 0; color: #273c75; }}
    p.lede {{ color: #2f3640; max-width: 960px; }}
    section {{ background: #fff; padding: 1rem; margin: 1rem 0; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); }}
    .figure {{ height: 100%; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class="lede">Interactive overview of a full year of lake sensor data (8 readings per day). Use the range slider, zoom, and hover to explore anomalies, seasonal shifts, and diurnal patterns.</p>
  </header>
  {figure_html}
</body>
</html>
"""
    return template


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize lake sensor readings.")
    parser.add_argument("csv", type=Path, help="Path to the readings CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lake_dashboard.html"),
        help="Destination HTML file (default: lake_dashboard.html)",
    )
    args = parser.parse_args()

    df = load_lake_data(args.csv)

    timeseries_fig = build_timeseries(df)
    heatmap_fig = build_hourly_heatmap(df, metric="temperature")
    monthly_fig = build_monthly_boxplots(df)
    corr_fig = build_correlation(df)

    html = build_dashboard_html(
        [timeseries_fig, heatmap_fig, monthly_fig, corr_fig],
        title="Lake sensor insights",
    )
    args.output.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
