"""Referral tracking, anti-fraud device/IP verification, and Option A reward fulfillment.

Heuristics strictly enforce anti-fraud protections against:
1. Self-referrals (same IP, same device fingerprint, or identical credentials).
2. Device fingerprint collisions (multi-accounting from the same browser/hardware).
3. IP address clustering (rate limiting referral creation per IP subnet/hash).
4. Delayed referrer qualification (Option A): Credits are only disbursed once the
   referee performs their first payment or processes 1,000+ billed inference tokens.
"""

import hashlib

from sqlalchemy import func
from sqlmodel import Session, select

from speedinfer.config import Settings, get_settings
from speedinfer.database.models import (
    ApiKey,
    PaymentTransaction,
    Referral,
    UsageLedger,
    User,
    utc_now,
)
from speedinfer.logging import get_logger

logger = get_logger("speedinfer.referrals")


def hash_ip(client_ip: str | None, pepper: str) -> str | None:
    """Compute secure HMAC-SHA256 hash of client IP address."""
    if not client_ip or not client_ip.strip():
        return None
    normalized_ip = client_ip.strip()
    return hashlib.sha256(f"{pepper}:{normalized_ip}".encode()).hexdigest()


def process_registration_referral(
    session: Session,
    new_user: User,
    referral_code: str | None,
    device_fingerprint: str | None,
    signup_ip_hash: str | None,
    settings: Settings | None = None,
) -> Referral | None:
    """Validate referral code, run anti-fraud checks, and create referral record.

    Args:
        session: Active database session.
        new_user: Newly created User entity.
        referral_code: Code supplied at signup.
        device_fingerprint: Client browser/hardware fingerprint.
        signup_ip_hash: Hashed client IP address.
        settings: Application settings.

    Returns:
        Referral | None: Created referral record if code was provided, else None.
    """
    if not referral_code or not referral_code.strip():
        return None

    cfg = settings or get_settings()
    clean_code = referral_code.strip().upper()

    referrer = session.exec(select(User).where(User.referral_code == clean_code)).first()
    if referrer is None:
        logger.warning(
            "Invalid referral code '%s' supplied for user %s",
            clean_code,
            new_user.email,
        )
        return None

    # Anti-Fraud Check 1: Direct Self-Referral
    if (
        referrer.id == new_user.id
        or referrer.email.lower() == new_user.email.lower()
        or (new_user.referral_code and clean_code == new_user.referral_code)
    ):
        logger.warning("Self-referral attempt blocked for user %s", new_user.email)
        ref = Referral(
            referrer_id=referrer.id,
            referred_id=new_user.id,
            status="rejected_fraud",
            reward_amount=cfg.referral_reward_amount,
            device_fingerprint=device_fingerprint,
            signup_ip_hash=signup_ip_hash,
            fraud_flag=True,
            fraud_reason="self_referral",
        )
        session.add(ref)
        return ref

    # Anti-Fraud Check 2: Same IP as Referrer
    if signup_ip_hash and referrer.signup_ip_hash and signup_ip_hash == referrer.signup_ip_hash:
        logger.warning(
            "Self-referral by IP match blocked: referrer %s, referee %s",
            referrer.id,
            new_user.id,
        )
        ref = Referral(
            referrer_id=referrer.id,
            referred_id=new_user.id,
            status="rejected_fraud",
            reward_amount=cfg.referral_reward_amount,
            device_fingerprint=device_fingerprint,
            signup_ip_hash=signup_ip_hash,
            fraud_flag=True,
            fraud_reason="self_referral",
        )
        session.add(ref)
        return ref

    # Anti-Fraud Check 3: Same Device Fingerprint as Referrer
    if (
        device_fingerprint
        and referrer.device_fingerprint
        and device_fingerprint == referrer.device_fingerprint
    ):
        logger.warning(
            "Self-referral by device fingerprint blocked: referrer %s, referee %s",
            referrer.id,
            new_user.id,
        )
        ref = Referral(
            referrer_id=referrer.id,
            referred_id=new_user.id,
            status="rejected_fraud",
            reward_amount=cfg.referral_reward_amount,
            device_fingerprint=device_fingerprint,
            signup_ip_hash=signup_ip_hash,
            fraud_flag=True,
            fraud_reason="self_referral",
        )
        session.add(ref)
        return ref

    # Anti-Fraud Check 0: Missing device fingerprint on referral registration in production
    if cfg.environment == "production" and (
        not device_fingerprint or not device_fingerprint.strip()
    ):
        logger.warning(
            "Referral attempt blocked: missing device fingerprint for user %s",
            new_user.email,
        )
        ref = Referral(
            referrer_id=referrer.id,
            referred_id=new_user.id,
            status="rejected_fraud",
            reward_amount=cfg.referral_reward_amount,
            device_fingerprint=None,
            signup_ip_hash=signup_ip_hash,
            fraud_flag=True,
            fraud_reason="missing_device_fingerprint",
        )
        session.add(ref)
        return ref

    # Anti-Fraud Check 4: Duplicate device fingerprint (multi-accounting)
    if device_fingerprint:
        existing_device_ref = session.exec(
            select(Referral).where(
                Referral.device_fingerprint == device_fingerprint,
                Referral.referred_id != new_user.id,
            )
        ).first()
        existing_device_user = session.exec(
            select(User).where(
                User.device_fingerprint == device_fingerprint,
                User.id != new_user.id,
            )
        ).first()
        if existing_device_ref or existing_device_user:
            logger.warning(
                "Multi-account referral farming blocked by device fingerprint: %s",
                device_fingerprint,
            )
            ref = Referral(
                referrer_id=referrer.id,
                referred_id=new_user.id,
                status="rejected_fraud",
                reward_amount=cfg.referral_reward_amount,
                device_fingerprint=device_fingerprint,
                signup_ip_hash=signup_ip_hash,
                fraud_flag=True,
                fraud_reason="duplicate_device",
            )
            session.add(ref)
            return ref

    # Anti-Fraud Check 5: IP clustering (max accounts or referrals from same IP)
    if signup_ip_hash:
        ip_ref_count = session.exec(
            select(func.count(Referral.id)).where(
                Referral.signup_ip_hash == signup_ip_hash,
                Referral.referred_id != new_user.id,
            )
        ).one()
        ip_user_count = session.exec(
            select(func.count(User.id)).where(
                User.signup_ip_hash == signup_ip_hash,
                User.id != new_user.id,
            )
        ).one()
        max_ip = getattr(cfg, "referral_max_accounts_per_ip", 3)
        if ip_ref_count >= 2 or ip_user_count >= max_ip:
            logger.warning(
                "Multi-account referral farming blocked by IP clustering: %s",
                signup_ip_hash,
            )
            ref = Referral(
                referrer_id=referrer.id,
                referred_id=new_user.id,
                status="rejected_fraud",
                reward_amount=cfg.referral_reward_amount,
                device_fingerprint=device_fingerprint,
                signup_ip_hash=signup_ip_hash,
                fraud_flag=True,
                fraud_reason="ip_clustering_limit",
            )
            session.add(ref)
            return ref

    # All anti-fraud checks passed: associate referral
    new_user.referred_by_id = referrer.id
    ref = Referral(
        referrer_id=referrer.id,
        referred_id=new_user.id,
        status="pending",
        reward_amount=cfg.referral_reward_amount,
        device_fingerprint=device_fingerprint,
        signup_ip_hash=signup_ip_hash,
        fraud_flag=False,
    )
    session.add(ref)
    session.add(new_user)
    return ref


def award_referee_bonus(
    session: Session,
    user: User,
    settings: Settings | None = None,
) -> bool:
    """Award +$5.00 extra trial credit to referee upon verified email confirmation.

    Args:
        session: Active database session.
        user: Verified referee user.
        settings: Application settings.

    Returns:
        bool: True if bonus was newly credited, False otherwise.
    """
    if not user.referred_by_id:
        return False

    cfg = settings or get_settings()
    referral = session.exec(select(Referral).where(Referral.referred_id == user.id)).first()
    if referral is None or referral.fraud_flag or referral.referee_bonus_awarded:
        return False

    bonus = cfg.referee_bonus_amount
    # Find active key or primary key
    key = session.exec(
        select(ApiKey)
        .where(ApiKey.user_id == user.id, ApiKey.is_active == True)  # noqa: E712
        .order_by(ApiKey.created_at.asc())
    ).first()

    if key is not None:
        key.credit_balance = round(key.credit_balance + bonus, 6)
        session.add(key)
        _sync_redis_balance(key.id, bonus)
    else:
        # User has no active key yet (e.g. signed up without default key).
        # Provision an initial active key with trial balance + referee bonus
        pepper = (
            cfg.api_key_pepper.get_secret_value()
            if hasattr(cfg.api_key_pepper, "get_secret_value")
            else str(cfg.api_key_pepper)
        )
        from speedinfer.core.auth import generate_api_key
        from speedinfer.database.models import TrialCreditGrant

        raw_key, prefix, key_hash = generate_api_key(pepper=pepper)
        initial_balance = round(cfg.trial_credit_balance + bonus, 6)
        new_key = ApiKey(
            user_id=user.id,
            name="default",
            key_hash=key_hash,
            prefix=prefix,
            permissions="chat:completions,completions,models:read,usage:read",
            credit_balance=initial_balance,
            rpm_limit=60,
            tpm_limit=60_000,
            is_active=True,
        )
        session.add(new_key)
        if session.get(TrialCreditGrant, user.id) is None:
            session.add(TrialCreditGrant(user_id=user.id, amount=initial_balance))

    referral.referee_bonus_awarded = True
    session.add(referral)
    return True


def check_and_award_referrer_bonus(
    session: Session,
    referee_user_id: int,
    settings: Settings | None = None,
) -> bool:
    """Enforce Option A: Award referrer bonus once referee pays or consumes 1,000+ tokens.

    Args:
        session: Active database session.
        referee_user_id: User identifier of the referee.
        settings: Application settings.

    Returns:
        bool: True if referrer reward was newly disbursed.
    """
    referee = session.get(User, referee_user_id)
    if referee is None or not referee.referred_by_id or referee.referral_reward_claimed:
        return False

    referral = session.exec(select(Referral).where(Referral.referred_id == referee.id)).first()
    if referral is None or referral.fraud_flag or referral.referrer_reward_awarded:
        return False

    # Check Qualification Criterion 1: Has referee made a payment?
    payment_count = session.exec(
        select(func.count(PaymentTransaction.id)).where(PaymentTransaction.user_id == referee.id)
    ).one()

    # Check Qualification Criterion 2: Has referee consumed >= 1,000 tokens?
    total_tokens = session.exec(
        select(func.coalesce(func.sum(UsageLedger.total_tokens), 0))
        .join(ApiKey, UsageLedger.api_key_id == ApiKey.id)
        .where(ApiKey.user_id == referee.id)
    ).one()

    if payment_count == 0 and total_tokens < 1000:
        return False

    cfg = settings or get_settings()
    # Disburse reward to referrer's primary active key
    referrer_key = session.exec(
        select(ApiKey)
        .where(ApiKey.user_id == referral.referrer_id, ApiKey.is_active == True)  # noqa: E712
        .order_by(ApiKey.created_at.asc())
    ).first()

    reward = referral.reward_amount
    if referrer_key is not None:
        referrer_key.credit_balance = round(referrer_key.credit_balance + reward, 6)
        session.add(referrer_key)
        _sync_redis_balance(referrer_key.id, reward)
    else:
        # Referrer has no active key; create default key funded with reward
        pepper = (
            cfg.api_key_pepper.get_secret_value()
            if hasattr(cfg.api_key_pepper, "get_secret_value")
            else str(cfg.api_key_pepper)
        )
        from speedinfer.core.auth import generate_api_key

        raw_key, prefix, key_hash = generate_api_key(pepper=pepper)
        referrer_key = ApiKey(
            user_id=referral.referrer_id,
            name="default",
            key_hash=key_hash,
            prefix=prefix,
            permissions="chat:completions,completions,models:read,usage:read",
            credit_balance=reward,
            rpm_limit=60,
            tpm_limit=60_000,
            is_active=True,
        )
        session.add(referrer_key)

    referral.status = "rewarded"
    referral.referrer_reward_awarded = True
    referral.rewarded_at = utc_now()
    referee.referral_reward_claimed = True

    session.add(referral)
    session.add(referee)
    session.flush()
    logger.info(
        "Option A referral bonus awarded: Referrer %s received $%s from Referee %s "
        "(tokens=%s, payments=%s)",
        referral.referrer_id,
        reward,
        referee.id,
        total_tokens,
        payment_count,
    )
    return True


def _sync_redis_balance(api_key_id: int, added_credits: float) -> None:
    """Increment cached balance in Redis if key is currently cached."""
    try:
        from speedinfer.gateway.redis import get_sync_redis

        redis_client = get_sync_redis()
        redis_key = f"speedinfer:balance:{api_key_id}"
        if redis_client.get(redis_key) is not None:
            redis_client.incrbyfloat(redis_key, added_credits)
    except Exception as exc:
        logger.debug("Redis balance sync skipped for key %s: %s", api_key_id, exc)
