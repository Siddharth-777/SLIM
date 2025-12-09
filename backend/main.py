#IMPORTS
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from supabase_client import SupabaseConfigError, get_supabase

#LOAD VARIABLES
load_dotenv()
API_KEY_ENV_VAR = "API_SECRET_KEY"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
TARGETS = ["ph", "turbidity", "temperature", "do_level"]


_tft_models: Dict[str, TemporalFusionTransformer] = {}
_tft_datasets: Dict[str, TimeSeriesDataSet] = {}

#API VERIFICATION
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    expected_key = os.getenv(API_KEY_ENV_VAR)
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{API_KEY_ENV_VAR} is not configured on the server",
        )
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

class LakeReading(BaseModel):
    ph: float
    turbidity: float
    temperature: float
    do_level: float


class ForecastResponse(BaseModel):
    forecast_timestamp: str = Field(
        ...,
        description="ISO8601 timestamp for the next predicted interval",
    )
    predictions: LakeReading


class LakeReadingResponse(LakeReading):
    id: int
    timestamp: Optional[str]

#API POINT
app = FastAPI(
    title="SLIM AI Lake Data API",
    docs_url="/data",
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_base_dataframe() -> pd.DataFrame:
    """Load and normalize the lake readings from the local CSV.

    The CSV is used instead of a request body so the endpoint can be hit
    without any inputs while still providing real data to the models.
    """

    data_path = Path(__file__).resolve().parent / "sample_lake_readings.csv"
    if not data_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data source not found at {data_path}",
        )

    df = pd.read_csv(data_path)

    if "timestamp" not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="CSV is missing the 'timestamp' column",
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["series_id"] = "buoy_1"
    df["time_idx"] = (
        (df["timestamp"] - df["timestamp"].min()).dt.total_seconds() // 3600
    ).astype(int)

    for column in TARGETS:
        if column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Column '{column}' missing from CSV",
            )
        df[column] = df[column].interpolate().bfill().ffill()

    return df


def _load_tft_resources(target: str) -> Tuple[TemporalFusionTransformer, TimeSeriesDataSet]:
    """Load cached TFT model + dataset definition for a target column."""

    if target not in TARGETS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown target '{target}'",
        )

    if target not in _tft_models or target not in _tft_datasets:
        ckpt_path = ARTIFACT_DIR / f"tft_{target}_best.ckpt"
        ds_path = ARTIFACT_DIR / f"tft_{target}_dataset.pkl"

        if not ckpt_path.exists() or not ds_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    f"Missing artifacts for {target}. Expected {ckpt_path.name} "
                    f"and {ds_path.name} inside {ARTIFACT_DIR}."
                ),
            )

        dataset = TimeSeriesDataSet.load(str(ds_path))
        model = TemporalFusionTransformer.load_from_checkpoint(
            checkpoint_path=str(ckpt_path),
            dataset=dataset,
            map_location=torch.device("cpu"),
        )
        model.eval()

        _tft_datasets[target] = dataset
        _tft_models[target] = model

    return _tft_models[target], _tft_datasets[target]


def _determine_step_hours(df: pd.DataFrame) -> int:
    """Infer the timestep spacing (in hours) from the dataset."""

    diffs = df["timestamp"].diff().dt.total_seconds().dropna()
    if diffs.empty:
        return 1
    # Use the most common step; default to 1 hour if inference fails
    mode_val = diffs.mode().iloc[0] if not diffs.mode().empty else 3600
    hours = max(1, int(round(mode_val / 3600)))
    return hours


def _prepare_prediction_frame(df: pd.DataFrame, prediction_length: int) -> Tuple[pd.DataFrame, str]:
    """Append future rows so the TFT model can forecast the next window."""

    step_hours = _determine_step_hours(df)
    last_timestamp = df["timestamp"].max()
    last_idx = int(df["time_idx"].max())

    future_rows = []
    for horizon in range(1, prediction_length + 1):
        future_rows.append(
            {
                "timestamp": last_timestamp + pd.Timedelta(hours=step_hours * horizon),
                "time_idx": last_idx + horizon,
                "series_id": "buoy_1",
                "ph": float("nan"),
                "turbidity": float("nan"),
                "temperature": float("nan"),
                "do_level": float("nan"),
            }
        )

    future_df = pd.DataFrame(future_rows)
    combined = pd.concat([df, future_df], ignore_index=True)
    forecast_timestamp = future_rows[0]["timestamp"].isoformat()
    return combined, forecast_timestamp


def _generate_forecast(df: pd.DataFrame) -> ForecastResponse:
    """Run TFT models for each parameter and return the next-step forecast."""

    predictions: Dict[str, float] = {}
    forecast_timestamp: Optional[str] = None

    for target in TARGETS:
        model, dataset = _load_tft_resources(target)
        prediction_length = dataset.max_prediction_length
        prepared_df, candidate_ts = _prepare_prediction_frame(df, prediction_length)

        predict_ds = TimeSeriesDataSet.from_dataset(
            dataset,
            prepared_df,
            predict=True,
            stop_randomization=True,
        )
        predict_loader = predict_ds.to_dataloader(train=False, batch_size=64, num_workers=0)

        # model.predict returns [batch, prediction_length]; we use the first horizon
        forecast_tensor = model.predict(predict_loader)
        if forecast_tensor.ndim == 3:
            # Take median quantile if quantile dimension is present
            forecast_tensor = forecast_tensor[..., forecast_tensor.shape[-1] // 2]
        next_value = float(forecast_tensor[0, 0].detach().cpu().item())

        predictions[target] = next_value
        forecast_timestamp = forecast_timestamp or candidate_ts

    return ForecastResponse(
        forecast_timestamp=forecast_timestamp,
        predictions=LakeReading(**predictions),
    )


#STORE READING
@app.post("/api/lake-data", status_code=status.HTTP_201_CREATED)
def ingest_lake_data(
    reading: LakeReading,
    _: None = Depends(verify_api_key),
):
    try:
        supabase = get_supabase()
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    response = supabase.table("lake_readings").insert(reading.model_dump()).execute()

    if getattr(response, "error", None):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(response.error),
        )

    return {"message": "Data received", "id": response.data[0].get("id")}

#FETCH LATEST READING
@app.get("/api/lake-data/latest", response_model=LakeReadingResponse)
def fetch_latest_reading():
    try:
        supabase = get_supabase()
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    response = (
        supabase.table("lake_readings")
        .select("id,timestamp,ph,turbidity,temperature,do_level")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if getattr(response, "error", None):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(response.error),
        )
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No readings available",
        )

    return response.data[0]


#FETCH READ HISTORY
@app.get("/api/lake-data/history", response_model=List[LakeReadingResponse])
def fetch_reading_history(limit: int = Query(100, gt=0, le=500)):
    try:
        supabase = get_supabase()
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    response = (
        supabase.table("lake_readings")
        .select("id,timestamp,ph,turbidity,temperature,do_level")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    if getattr(response, "error", None):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(response.error),
        )
    return response.data


@app.get("/forecast/all", response_model=ForecastResponse)
def forecast_all():
    """Forecast all four lake parameters using TFT models stored in artifacts/.

    The endpoint does not require a request body. Instead, it loads the latest
    readings from the repository CSV and feeds them into the saved TFT models to
    predict the next timestep.
    """

    base_df = _load_base_dataframe()
    return _generate_forecast(base_df)
