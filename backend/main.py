import os
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from supabase_client import SupabaseConfigError, get_supabase


API_KEY_ENV_VAR = "API_SECRET_KEY"


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
    ph: float = Field(..., description="pH level of the water")
    turbidity: float = Field(..., description="Turbidity measurement")
    temperature: float = Field(..., description="Temperature in Celsius")
    do_level: float = Field(..., description="Dissolved oxygen level")


class LakeReadingResponse(LakeReading):
    id: int
    timestamp: Optional[str] = Field(None, description="ISO timestamp of the reading")


app = FastAPI(title="SLIM AI Lake Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/lake-data", status_code=status.HTTP_201_CREATED)
def ingest_lake_data(
    reading: LakeReading,
    _: None = Depends(verify_api_key),
):
    try:
        supabase = get_supabase()
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    response = (
        supabase.table("lake_readings")
        .insert(reading.dict())
        .execute()
    )

    if response.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.error.message,
        )

    inserted = response.data[0]
    return {"message": "Data received", "id": inserted.get("id")}


@app.get("/api/lake-data/latest", response_model=LakeReadingResponse)
def fetch_latest_reading():
    try:
        supabase = get_supabase()
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    response = (
        supabase.table("lake_readings")
        .select("id,timestamp,ph,turbidity,temperature,do_level")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )

    if response.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.error.message,
        )

    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No readings available"
        )

    return response.data[0]


@app.get("/api/lake-data/history", response_model=List[LakeReadingResponse])
def fetch_reading_history(limit: int = Query(100, gt=0, le=500)):
    try:
        supabase = get_supabase()
    except SupabaseConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    response = (
        supabase.table("lake_readings")
        .select("id,timestamp,ph,turbidity,temperature,do_level")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )

    if response.error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.error.message,
        )

    return response.data


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
