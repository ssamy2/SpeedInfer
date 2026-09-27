"""Referral program endpoints and dashboard analytics."""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlmodel import Session, select

from speedinfer.config import Settings, get_settings
from speedinfer.core.security import get_current_user
from speedinfer.database.models import Referral, User
from speedinfer.database.session import get_session
from speedinfer.gateway.schemas import ReferralStatsResponse

router = APIRouter(prefix="/v1/referrals", tags=["Referrals"])


@router.get(
    "/me",
    response_model=ReferralStatsResponse,
    summary="Get authenticated user's referral code and metrics",
    description=(
        "Returns personal referral code, shareable invite link, "
        "conversion counts, and cumulative earnings."
    ),
)
async def get_my_referrals(
    current_user: Annotated[User, Depends(get_current_user)],
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReferralStatsResponse:
    if not current_user.referral_code:
        from speedinfer.database.models import generate_referral_code

        current_user.referral_code = generate_referral_code()
        session.add(current_user)
        session.commit()
        session.refresh(current_user)

    code = current_user.referral_code
    base_url = settings.public_base_url.rstrip("/")
    link = f"{base_url}/?ref={code}"

    # Query referral counts
    total_refs = session.exec(
        select(func.count(Referral.id)).where(Referral.referrer_id == current_user.id)
    ).one()

    pending_refs = session.exec(
        select(func.count(Referral.id)).where(
            Referral.referrer_id == current_user.id,
            Referral.status == "pending",
            Referral.fraud_flag == False,  # noqa: E712
        )
    ).one()

    converted_refs = session.exec(
        select(func.count(Referral.id)).where(
            Referral.referrer_id == current_user.id,
            Referral.referrer_reward_awarded == True,  # noqa: E712
        )
    ).one()

    total_earnings = session.exec(
        select(func.coalesce(func.sum(Referral.reward_amount), 0.0)).where(
            Referral.referrer_id == current_user.id,
            Referral.referrer_reward_awarded == True,  # noqa: E712
        )
    ).one()

    return ReferralStatsResponse(
        referral_code=code,
        referral_link=link,
        total_referrals=int(total_refs),
        pending_referrals=int(pending_refs),
        converted_referrals=int(converted_refs),
        total_earnings_usd=round(float(total_earnings), 2),
        reward_per_referral_usd=settings.referral_reward_amount,
        referee_bonus_usd=settings.referee_bonus_amount,
    )
