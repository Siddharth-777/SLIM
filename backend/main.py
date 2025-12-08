#IMPORTS
import os
from typing import List, Optional
from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase_client import SupabaseConfigError, get_supabase

#LOAD VARIABLES
load_dotenv()
API_KEY_ENV_VAR = "API_SECRET_KEY"

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