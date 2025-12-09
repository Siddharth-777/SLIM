# tft_api_server.py
# FastAPI server that uses the 4 trained TFT models
# to forecast future pH, turbidity, temperature and DO.

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


# ---------------------------------------------------------
# Paths / config
# ---------------------------------------------------------


BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
CSV_PATH = BASE_DIR / "sample_lake_readings.csv"

# These names MUST match what train_tft_lake_multi.py saved
TARGETS: Dict[str, Dict[str, Path]] = {
    "ph": {
        "dataset": ARTIFACT_DIR / "tft_ph_dataset.pkl",
        "ckpt": ARTIFACT_DIR / "tft_ph_best.ckpt",
    },
    "turbidity": {
        "dataset": ARTIFACT_DIR / "tft_turbidity_dataset.pkl",
        "ckpt": ARTIFACT_DIR / "tft_turbidity_best.ckpt",
    },
    "temperature": {
        "dataset": ARTIFACT_DIR / "tft_temperature_dataset.pkl",
        "ckpt": ARTIFACT_DIR / "tft_temperature_best.ckpt",
    },
    "do_level": {
        "dataset": ARTIFACT_DIR / "tft_do_level_dataset.pkl",
        "ckpt": ARTIFACT_DIR / "tft_do_level_best.ckpt",
    },
}


# ---------------------------------------------------------
# Pydantic models for response
# ---------------------------------------------------------


class ForecastPoint(BaseModel):
    timestamp: datetime
    median: float
    p10: float
    p90: float


class VariableForecast(BaseModel):
    name: str = Field(..., description="Target variable name (ph, turbidity, temperature, do_level)")
    horizon_steps: int
    forecast: List[ForecastPoint]


class MultiForecastResponse(BaseModel):
    step_hours: float = Field(..., description="Average step size in hours between readings")
    variables: List[VariableForecast]


# ---------------------------------------------------------
# Load artifacts at startup
# ---------------------------------------------------------


def load_artifacts() -> Dict[str, Tuple[TimeSeriesDataSet, TemporalFusionTransformer]]:
    models: Dict[str, Tuple[TimeSeriesDataSet, TemporalFusionTransformer]] = {}

    for target, paths in TARGETS.items():
        ds_path = paths["dataset"]
        ckpt_path = paths["ckpt"]

        if not ds_path.exists():
            raise RuntimeError(f"Dataset file not found for '{target}': {ds_path}")
        if not ckpt_path.exists():
            raise RuntimeError(f"Checkpoint file not found for '{target}': {ckpt_path}")

        dataset: TimeSeriesDataSet = TimeSeriesDataSet.load(str(ds_path))

        model = TemporalFusionTransformer.load_from_checkpoint(
            str(ckpt_path),
            map_location=torch.device("cpu"),
        )
        model.eval()

        models[target] = (dataset, model)

    return models


try:
    TARGET_MODELS = load_artifacts()
    print("✅ Loaded TFT artifacts for targets:", list(TARGET_MODELS.keys()))
except Exception as e:
    print("❌ Error loading TFT artifacts:", repr(e))
    TARGET_MODELS = {}  # keep empty so we can error nicely via API


# ---------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------


def load_history_from_csv() -> pd.DataFrame:
    """
    Load the same CSV used for training, build time_idx etc.
    This is your 'recent history' source.
    """
    if not CSV_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"CSV file not found at {CSV_PATH}.",
        )

    df = pd.read_csv(CSV_PATH)

    if "timestamp" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail="CSV missing 'timestamp' column.",
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # series id & time_idx must match training logic
    df["series_id"] = "buoy_1"
    df["time_idx"] = (
        (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
    ).astype(int)

    # interpolate/fill numeric columns
    for col in ["ph", "turbidity", "temperature", "do_level"]:
        if col not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"CSV missing '{col}' column.",
            )
        df[col] = df[col].interpolate().bfill().ffill()

    return df


def compute_step_hours(df: pd.DataFrame) -> float:
    """Median step size in hours, based on timestamp diffs."""
    if len(df) < 2:
        return 1.0
    diffs = df["timestamp"].diff().dropna().dt.total_seconds() / 3600.0
    if len(diffs) == 0:
        return 1.0
    return float(diffs.median())


# ---------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------


def infer_for_target(
    target_name: str,
    df_full: pd.DataFrame,
) -> Tuple[float, List[ForecastPoint]]:
    """
    Run TFT for a single target over the recent window
    and return (step_hours, forecast_points).
    """
    if target_name not in TARGET_MODELS:
        raise HTTPException(
            status_code=500,
            detail=f"Model for target '{target_name}' is not loaded.",
        )

    dataset, model = TARGET_MODELS[target_name]

    encoder_len: int = dataset.max_encoder_length
    pred_len: int = dataset.max_prediction_length

    if len(df_full) < encoder_len:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Not enough rows in CSV for '{target_name}': "
                f"need at least {encoder_len}, got {len(df_full)}"
            ),
        )

    # use only the last encoder_len rows as the 'recent history' window
    df_recent = df_full.iloc[-encoder_len:].copy()

    # make dataset compatible with training dataset
    new_dataset = TimeSeriesDataSet.from_dataset(
        dataset,
        df_recent,
        stop_randomization=True,
    )

    loader = new_dataset.to_dataloader(
        train=False,
        batch_size=1,
        num_workers=0,
    )

    with torch.no_grad():
        # Some versions return (raw_predictions, x),
        # others (raw_predictions, x, index, ...) etc.
        pred_result = model.predict(
            loader,
            mode="raw",
            return_x=True,
        )

    if isinstance(pred_result, tuple):
        raw_predictions = pred_result[0]
    else:
        raw_predictions = pred_result

    # raw_predictions["prediction"]: [batch, prediction_length, num_quantiles]
    preds = raw_predictions["prediction"][0]  # first (and only) sample
    preds_np = preds.detach().cpu().numpy()

    quantiles = list(model.loss.quantiles)

    def q_index(q: float, default_idx: int) -> int:
        return quantiles.index(q) if q in quantiles else default_idx

    q10_idx = q_index(0.1, 1)
    q50_idx = q_index(0.5, len(quantiles) // 2)
    q90_idx = q_index(0.9, -2)

    p10 = preds_np[:, q10_idx]
    p50 = preds_np[:, q50_idx]
    p90 = preds_np[:, q90_idx]

    horizon = preds_np.shape[0]  # should be pred_len

    # infer step size from the CSV timestamps
    step_hours = compute_step_hours(df_recent)
    last_ts = df_recent["timestamp"].iloc[-1]

    points: List[ForecastPoint] = []
    for i in range(horizon):
        ts = last_ts + timedelta(hours=step_hours * (i + 1))
        points.append(
            ForecastPoint(
                timestamp=ts,
                median=float(p50[i]),
                p10=float(p10[i]),
                p90=float(p90[i]),
            )
        )

    return step_hours, points


# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------


app = FastAPI(
    title="Lake Multi-Parameter Forecast API",
    version="1.0.0",
    description=(
        "Forecast future pH, turbidity, temperature and DO levels "
        "using TFT models trained from sample_lake_readings.csv."
    ),
)


@app.get("/api/forecast/all", response_model=MultiForecastResponse)
def forecast_all():
    """
    Forecast future pH, turbidity, temperature and DO levels.

    - Uses recent history from sample_lake_readings.csv
    - Builds encoder window per target
    - Runs each of the 4 TFT models
    """
    if not TARGET_MODELS:
        raise HTTPException(
            status_code=500,
            detail="No TFT models loaded at startup. Check server logs.",
        )

    try:
        df = load_history_from_csv()
        variables: List[VariableForecast] = []
        common_step_hours: float | None = None

        for target_name in TARGETS.keys():
            step_hours, points = infer_for_target(target_name, df)

            if common_step_hours is None:
                common_step_hours = step_hours

            variables.append(
                VariableForecast(
                    name=target_name,
                    horizon_steps=len(points),
                    forecast=points,
                )
            )

        return MultiForecastResponse(
            step_hours=common_step_hours if common_step_hours is not None else 1.0,
            variables=variables,
        )

    except HTTPException:
        # re-raise clean HTTP errors
        raise
    except Exception as e:
        # log + return readable message instead of plain "Internal Server Error"
        print("❌ Error during /api/forecast/all:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while generating forecast: {e}",
        )
