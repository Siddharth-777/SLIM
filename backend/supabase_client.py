import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()


class SupabaseConfigError(RuntimeError):
    """Raised when Supabase configuration is invalid."""


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    """Return a cached Supabase client configured from environment variables."""
    url: Optional[str] = os.getenv("SUPABASE_URL")
    service_role_key: Optional[str] = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not service_role_key:
        raise SupabaseConfigError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in the environment"
        )

    return create_client(url, service_role_key)
