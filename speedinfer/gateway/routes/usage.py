"""Usage and credit balance inspection endpoint.

Provides:
- GET /v1/usage
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from speedinfer.core.auth import require_scope
from speedinfer.database.models import ApiKey
from speedinfer.gateway.schemas import UsageResponse

router = APIRouter(prefix="/v1", tags=["Usage"])


@router.get(
    "/usage",
    response_model=UsageResponse,
    summary="Get current API key credit balance and quotas",
    description="Returns remaining credit balance, rate limits, and currency denomination.",
)
async def get_usage(
    api_key: Annotated[ApiKey, Depends(require_scope("usage:read"))],
) -> UsageResponse:
    """Retrieve authenticated API key balance and configured rate limits."""
    return UsageResponse(
        credit_balance=round(float(api_key.credit_balance), 6),
        balance=round(float(api_key.credit_balance), 6),
        paid_balance=round(float(getattr(api_key, "paid_balance", 0.0)), 6),
        trial_balance=round(float(getattr(api_key, "trial_balance", 0.0)), 6),
        currency="USD",
        rpm_limit=api_key.rpm_limit,
        tpm_limit=api_key.tpm_limit,
    )
