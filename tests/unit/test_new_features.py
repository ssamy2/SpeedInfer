"""Comprehensive unit and integration tests for Turnstile, anti-fraud referrals,
email OTP verification, password reset, OAuth provisioning, and profile settings.
"""

from datetime import timedelta
from unittest.mock import patch

import httpx
import pytest
from pydantic import SecretStr
from sqlmodel import Session, select

from speedinfer.config import Settings
from speedinfer.core.referrals import (
    award_referee_bonus,
    check_and_award_referrer_bonus,
    hash_ip,
    process_registration_referral,
)
from speedinfer.core.security import hash_password, verify_password
from speedinfer.core.turnstile import verify_turnstile_token
from speedinfer.database.models import (
    ApiKey,
    EmailVerificationCode,
    OAuthAccount,
    PaymentTransaction,
    Referral,
    User,
    utc_now,
)
from speedinfer.gateway.routes.oauth import _provision_oauth_user


# ---------------------------------------------------------------------------
# 1. Cloudflare Turnstile Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_turnstile_bypass_when_disabled() -> None:
    """Verify Turnstile verification passes immediately when disabled."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        turnstile_enabled=False,
    )
    result = await verify_turnstile_token(None, settings=settings)
    assert result is True

    result_token = await verify_turnstile_token("any-token-string", settings=settings)
    assert result_token is True


@pytest.mark.asyncio
async def test_turnstile_mock_tokens() -> None:
    """Verify internal mock development tokens pass or fail accurately."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        turnstile_enabled=True,
    )
    # Success mock token
    res_ok = await verify_turnstile_token("turnstile-mock-token-ok", settings=settings)
    assert res_ok is True

    # Cloudflare dummy 1x00000000000000000000AA always passes
    res_cf_dummy = await verify_turnstile_token("1x00000000000000000000AA", settings=settings)
    assert res_cf_dummy is True

    # Failure mock token
    with pytest.raises(Exception) as exc_info:
        await verify_turnstile_token("turnstile-mock-token-fail", settings=settings)
    assert "Security verification failed" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_turnstile_missing_token_when_enabled() -> None:
    """Verify missing token is rejected with 400 Bad Request."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        turnstile_enabled=True,
        environment="production",
    )
    with pytest.raises(Exception) as exc_info:
        await verify_turnstile_token(None, settings=settings)
    assert "Security verification (CAPTCHA) token is required" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_turnstile_api_siteverify_mocked() -> None:
    """Verify siteverify API response parsing with action and hostname checks."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        turnstile_enabled=True,
        environment="production",
        turnstile_secret_key=SecretStr("mock-secret-key"),
        turnstile_hostnames="speedinfer.com,localhost",
    )

    # 1. Success response
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = httpx.Response(
            status_code=200,
            json={"success": True, "action": "login", "hostname": "localhost"},
            request=httpx.Request(
                "POST", "https://challenges.cloudflare.com/turnstile/v0/siteverify"
            ),
        )
        mock_post.return_value = mock_resp

        verified = await verify_turnstile_token(
            "real-cf-token",
            expected_action="login",
            settings=settings,
        )
        assert verified is True

    # 2. Action mismatch
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = httpx.Response(
            status_code=200,
            json={"success": True, "action": "contact", "hostname": "localhost"},
            request=httpx.Request(
                "POST", "https://challenges.cloudflare.com/turnstile/v0/siteverify"
            ),
        )
        mock_post.return_value = mock_resp

        with pytest.raises(Exception) as exc_info:
            await verify_turnstile_token(
                "real-cf-token", expected_action="login", settings=settings
            )
        assert "action mismatch" in str(exc_info.value.detail).lower()

    # 3. Hostname mismatch
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = httpx.Response(
            status_code=200,
            json={"success": True, "action": "login", "hostname": "attacker-domain.com"},
            request=httpx.Request(
                "POST", "https://challenges.cloudflare.com/turnstile/v0/siteverify"
            ),
        )
        mock_post.return_value = mock_resp

        with pytest.raises(Exception) as exc_info:
            await verify_turnstile_token(
                "real-cf-token", expected_action="login", settings=settings
            )
        assert "not authorized" in str(exc_info.value.detail).lower()


# ---------------------------------------------------------------------------
# 2. Referral Anti-Fraud & Attribution Tests
# ---------------------------------------------------------------------------
def test_referral_anti_fraud_duplicate_device(db_session: Session) -> None:
    """Verify duplicate device fingerprints from the same device are flagged as fraud."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        referral_anti_fraud_enabled=True,
    )
    pepper = "test-pepper-32-chars-minimum-len"

    # Referrer
    referrer = User(email="referrer1@speedinfer.com", referral_code="REF-DEV1")
    db_session.add(referrer)
    db_session.commit()
    db_session.refresh(referrer)

    # First user on device
    user1 = User(
        email="device_user1@speedinfer.com",
        device_fingerprint="device-fingerprint-abc-123",
        signup_ip_hash=hash_ip("192.168.1.1", pepper),
    )
    db_session.add(user1)
    db_session.commit()
    db_session.refresh(user1)

    # Second user on SAME device attempting to use referral
    user2 = User(
        email="device_user2@speedinfer.com",
        device_fingerprint="device-fingerprint-abc-123",
        signup_ip_hash=hash_ip("192.168.1.2", pepper),
    )
    db_session.add(user2)
    db_session.commit()
    db_session.refresh(user2)

    process_registration_referral(
        session=db_session,
        new_user=user2,
        referral_code=referrer.referral_code,
        device_fingerprint="device-fingerprint-abc-123",
        signup_ip_hash=hash_ip("192.168.1.2", pepper),
        settings=settings,
    )
    db_session.commit()

    # User 2 should NOT be linked to referrer
    assert user2.referred_by_id is None
    # Fraud record must be registered
    ref_record = db_session.exec(select(Referral).where(Referral.referred_id == user2.id)).first()
    assert ref_record is not None
    assert ref_record.fraud_flag is True
    assert ref_record.fraud_reason == "duplicate_device"


def test_referral_anti_fraud_ip_clustering(db_session: Session) -> None:
    """Verify IP clustering threshold blocks mass account creation on the same IP."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        referral_anti_fraud_enabled=True,
        referral_max_accounts_per_ip=3,
    )
    pepper = "test-pepper-32-chars-minimum-len"
    ip_hash = hash_ip("203.0.113.42", pepper)

    referrer = User(email="cluster_ref@speedinfer.com", referral_code="REF-CLUSTER")
    db_session.add(referrer)
    db_session.commit()

    # Create 3 existing accounts on same IP
    for i in range(3):
        u = User(
            email=f"ip_farm_{i}@speedinfer.com",
            signup_ip_hash=ip_hash,
            device_fingerprint=f"unique_dev_{i}",
        )
        db_session.add(u)
    db_session.commit()

    # 4th account from same IP
    user_4 = User(
        email="ip_farm_4@speedinfer.com",
        signup_ip_hash=ip_hash,
        device_fingerprint="unique_dev_4",
    )
    db_session.add(user_4)
    db_session.commit()

    process_registration_referral(
        session=db_session,
        new_user=user_4,
        referral_code="REF-CLUSTER",
        device_fingerprint="unique_dev_4",
        signup_ip_hash=ip_hash,
        settings=settings,
    )
    db_session.commit()

    assert user_4.referred_by_id is None
    ref_record = db_session.exec(select(Referral).where(Referral.referred_id == user_4.id)).first()
    assert ref_record is not None
    assert ref_record.fraud_flag is True
    assert ref_record.fraud_reason == "ip_clustering_limit"


def test_referral_anti_fraud_self_referral(db_session: Session) -> None:
    """Verify self-referral attempts are rejected and flagged."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        referral_anti_fraud_enabled=True,
    )
    pepper = "test-pepper-32-chars-minimum-len"

    user = User(
        email="self_ref@speedinfer.com",
        referral_code="MY-OWN-CODE",
        device_fingerprint="dev-self",
        signup_ip_hash=hash_ip("1.1.1.1", pepper),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    process_registration_referral(
        session=db_session,
        new_user=user,
        referral_code="MY-OWN-CODE",
        device_fingerprint="dev-self",
        signup_ip_hash=hash_ip("1.1.1.1", pepper),
        settings=settings,
    )
    db_session.commit()

    assert user.referred_by_id is None
    ref_record = db_session.exec(select(Referral).where(Referral.referred_id == user.id)).first()
    assert ref_record is not None
    assert ref_record.fraud_flag is True
    assert ref_record.fraud_reason == "self_referral"


# ---------------------------------------------------------------------------
# 3. Option A Bonus Qualification & Disbursement Tests
# ---------------------------------------------------------------------------
def test_option_a_referee_bonus_and_delayed_referrer_reward(db_session: Session) -> None:
    """Verify Option A behavior:
    1. Referee receives +$5.00 upon email verification.
    2. Referrer receives NOTHING until referee reaches 1,000 billed tokens or payment.
    3. Referrer gets +$5.00 once qualification condition is met.
    """
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        referral_reward_amount=2.00,
        referee_bonus_amount=0.00,
        referral_min_topup=20.00,
    )

    # 1. Create Referrer with an API key
    referrer = User(email="referrer_a@speedinfer.com", referral_code="REFA100")
    db_session.add(referrer)
    db_session.commit()
    db_session.refresh(referrer)

    ref_key = ApiKey(
        user_id=referrer.id,
        name="referrer-key",
        key_hash="a" * 64,
        prefix="sk-speedinfer-12345678",
        credit_balance=10.00,
    )
    db_session.add(ref_key)
    db_session.commit()

    # 2. Referee registers with referrer's code
    referee = User(
        email="referee_a@speedinfer.com",
        device_fingerprint="dev_referee_unique",
        signup_ip_hash="hash_ip_unique",
    )
    db_session.add(referee)
    db_session.commit()
    db_session.refresh(referee)

    referee_key = ApiKey(
        user_id=referee.id,
        name="referee-key",
        key_hash="b" * 64,
        prefix="sk-speedinfer-87654321",
        credit_balance=0.00,
    )
    db_session.add(referee_key)
    db_session.commit()

    # Process legitimate referral
    process_registration_referral(
        session=db_session,
        new_user=referee,
        referral_code=referrer.referral_code,
        device_fingerprint="dev_referee_unique",
        signup_ip_hash="hash_ip_unique",
        settings=settings,
    )
    db_session.commit()

    assert referee.referred_by_id == referrer.id
    ref_record = db_session.exec(select(Referral).where(Referral.referred_id == referee.id)).first()
    assert ref_record is not None
    assert ref_record.status == "pending"
    assert ref_record.referrer_reward_awarded is False

    # 3. Referee verifies email -> Referee bonus check (0 configured default bonus)
    awarded = award_referee_bonus(db_session, referee, settings=settings)
    db_session.commit()
    assert awarded is True
    db_session.refresh(referee_key)
    assert referee_key.credit_balance == 0.00

    # Referrer should NOT yet be rewarded
    db_session.refresh(ref_key)
    assert ref_key.credit_balance == 10.00

    # 4. Referee tops up $15.00 -> below $20.00 threshold, no referrer reward
    tx1 = PaymentTransaction(
        provider="whop",
        provider_payment_id="whop_tx_below_threshold",
        webhook_id="webhook_tx_1",
        user_id=referee.id,
        api_key_id=referee_key.id,
        amount_usd=15.0,
        credits_added=15.0,
    )
    db_session.add(tx1)
    db_session.commit()

    awarded_ref = check_and_award_referrer_bonus(db_session, referee.id, settings=settings)
    db_session.commit()
    assert awarded_ref is False
    db_session.refresh(ref_key)
    assert ref_key.credit_balance == 10.00

    # 5. Referee tops up $25.00 (> $20.00) -> qualifies for $2.00 referrer reward!
    tx2 = PaymentTransaction(
        provider="whop",
        provider_payment_id="whop_tx_qualifying",
        webhook_id="webhook_tx_2",
        user_id=referee.id,
        api_key_id=referee_key.id,
        amount_usd=25.0,
        credits_added=25.0,
    )
    db_session.add(tx2)
    db_session.commit()

    awarded_ref = check_and_award_referrer_bonus(db_session, referee.id, settings=settings)
    db_session.commit()
    assert awarded_ref is True

    # Referrer receives +$2.00 in trial_balance!
    db_session.refresh(ref_key)
    assert ref_key.trial_balance == 2.00
    assert ref_key.paid_balance == 10.00
    assert ref_key.credit_balance == 12.00
    db_session.refresh(ref_record)
    assert ref_record.status == "rewarded"
    assert ref_record.referrer_reward_awarded is True

    # 6. Idempotency: subsequent check does not double reward
    awarded_again = check_and_award_referrer_bonus(db_session, referee.id, settings=settings)
    assert awarded_again is False
    db_session.refresh(ref_key)
    assert ref_key.credit_balance == 12.00


# ---------------------------------------------------------------------------
# 4. Email Verification OTP & Password Reset Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_email_verification_and_resend_cooldown(db_session: Session) -> None:
    """Verify email verification OTP flow and 60-second cooldown enforcement."""
    from speedinfer.gateway.routes.auth import resend_code, verify_email
    from speedinfer.gateway.schemas import ResendCodeRequest, VerifyEmailRequest

    user = User(email="verify_test@speedinfer.com", is_verified=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # 1. Send OTP code
    otp = EmailVerificationCode(
        email=user.email,
        code="654321",
        purpose="registration",
        expires_at=utc_now() + timedelta(minutes=15),
    )
    db_session.add(otp)
    db_session.commit()

    # 2. Resend code immediately -> must fail with 429 Too Many Requests
    with pytest.raises(Exception) as exc_info:
        await resend_code(
            ResendCodeRequest(email=user.email, purpose="registration"),
            session=db_session,
        )
    assert exc_info.value.status_code == 429

    # 3. Invalid code -> fails 400
    with pytest.raises(Exception) as exc_info:
        await verify_email(
            VerifyEmailRequest(email=user.email, code="000000"),
            session=db_session,
        )
    assert exc_info.value.status_code == 400

    # 4. Valid code -> success
    result = await verify_email(
        VerifyEmailRequest(email=user.email, code="654321"),
        session=db_session,
    )
    assert result["is_verified"] is True
    db_session.refresh(user)
    assert user.is_verified is True


@pytest.mark.asyncio
async def test_password_reset_flow(db_session: Session) -> None:
    """Verify full forgot-password -> reset-password flow."""
    from speedinfer.gateway.routes.auth import reset_password
    from speedinfer.gateway.schemas import ResetPasswordRequest

    initial_pwd_hash = hash_password("OldPassword123!")
    user = User(email="forgot_user@speedinfer.com", password_hash=initial_pwd_hash, is_active=True)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Issue reset code
    code_record = EmailVerificationCode(
        email=user.email,
        code="998877",
        purpose="password_reset",
        expires_at=utc_now() + timedelta(minutes=15),
    )
    db_session.add(code_record)
    db_session.commit()

    # Reset password with correct OTP
    reset_req = ResetPasswordRequest(
        email=user.email,
        code="998877",
        new_password="NewSuperSecurePassword456!",
    )
    resp = await reset_password(reset_req, session=db_session)
    assert "Password has been reset successfully" in resp["message"]

    db_session.refresh(user)
    assert verify_password("NewSuperSecurePassword456!", user.password_hash) is True
    assert verify_password("OldPassword123!", user.password_hash) is False

    # Attempting to reuse the code must fail
    with pytest.raises(Exception) as exc_info:
        await reset_password(reset_req, session=db_session)
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# 5. Profile Settings Tests (GET & PATCH /v1/auth/profile)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_user_profile_get_and_patch(db_session: Session) -> None:
    """Verify user can inspect and update their avatar, address/location, and name."""
    from speedinfer.gateway.routes.auth import get_profile, update_profile
    from speedinfer.gateway.schemas import ProfileUpdateRequest

    user = User(
        email="profile_dev@speedinfer.com",
        name="Developer Initial",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # 1. GET profile
    profile = await get_profile(current_user=user, session=db_session)
    assert profile.email == "profile_dev@speedinfer.com"
    assert profile.name == "Developer Initial"
    assert profile.avatar_url is None

    # 2. PATCH profile
    update_req = ProfileUpdateRequest(
        name="Dr. Alan Turing",
        avatar_url="https://speedinfer.com/avatars/alan.png",
        location="Cairo, Egypt",
        organization="SpeedInfer Research",
    )
    updated = await update_profile(payload=update_req, current_user=user, session=db_session)
    assert updated.name == "Dr. Alan Turing"
    assert updated.avatar_url == "https://speedinfer.com/avatars/alan.png"
    assert updated.location == "Cairo, Egypt"
    assert updated.organization == "SpeedInfer Research"

    db_session.refresh(user)
    assert user.avatar_url == "https://speedinfer.com/avatars/alan.png"
    assert user.location == "Cairo, Egypt"


# ---------------------------------------------------------------------------
# 6. OAuth User Provisioning Tests
# ---------------------------------------------------------------------------
def test_oauth_provisioning_and_linking(db_session: Session) -> None:
    """Verify OAuth provisioning creates users, links accounts, and honors referrals."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
    )

    referrer = User(email="oauth_ref@speedinfer.com", referral_code="REF-OAUTH")
    db_session.add(referrer)
    db_session.commit()

    # 1. Brand new user via Google
    user_google = _provision_oauth_user(
        session=db_session,
        provider="google",
        provider_user_id="google-uid-1001",
        email="google_user@gmail.com",
        name="Google Developer",
        avatar_url="https://lh3.googleusercontent.com/photo.jpg",
        referral_code="REF-OAUTH",
        settings=settings,
    )
    assert user_google.id is not None
    assert user_google.email == "google_user@gmail.com"
    assert user_google.avatar_url == "https://lh3.googleusercontent.com/photo.jpg"
    assert user_google.is_verified is True
    assert user_google.referred_by_id == referrer.id

    # Verify referee starts with 0.00 default balance on default API key
    google_key = db_session.exec(select(ApiKey).where(ApiKey.user_id == user_google.id)).first()
    assert google_key is not None
    assert google_key.credit_balance == 0.00
    assert google_key.trial_balance == 0.00
    assert google_key.paid_balance == 0.00
    ref_row = db_session.exec(
        select(Referral).where(Referral.referred_id == user_google.id)
    ).first()
    assert ref_row is not None
    assert ref_row.referee_bonus_awarded is True

    # 2. Same Google user signs in again -> retrieves existing user
    user_google_2 = _provision_oauth_user(
        session=db_session,
        provider="google",
        provider_user_id="google-uid-1001",
        email="google_user@gmail.com",
        name="Google Developer",
        avatar_url="https://lh3.googleusercontent.com/photo.jpg",
        referral_code=None,
        settings=settings,
    )
    assert user_google_2.id == user_google.id

    # 3. Existing email links GitHub account
    user_github = _provision_oauth_user(
        session=db_session,
        provider="github",
        provider_user_id="github-uid-2002",
        email="google_user@gmail.com",
        name="GitHub Developer",
        avatar_url=None,
        referral_code=None,
        settings=settings,
    )
    assert user_github.id == user_google.id

    # Verify both OAuthAccount rows linked to same user
    oauth_rows = db_session.exec(
        select(OAuthAccount).where(OAuthAccount.user_id == user_google.id)
    ).all()
    assert len(oauth_rows) == 2


# ---------------------------------------------------------------------------
# 7. Referral Dashboard Endpoint (`GET /v1/referrals/me`)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_referral_dashboard_endpoint(db_session: Session) -> None:
    """Verify referral metrics API returns accurate conversion stats."""
    from speedinfer.gateway.routes.referrals import get_my_referrals

    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        referral_reward_amount=5.00,
        referee_bonus_amount=5.00,
        public_base_url="https://speedinfer.com",
    )

    user = User(email="stats_user@speedinfer.com", referral_code="STATS-101")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Create 2 referee users to satisfy foreign key constraints
    u1 = User(email="referee_user1@speedinfer.com")
    u2 = User(email="referee_user2@speedinfer.com")
    db_session.add(u1)
    db_session.add(u2)
    db_session.commit()

    # Add 2 referrals: 1 qualified ($5.00), 1 pending
    ref1 = Referral(
        referrer_id=user.id,
        referred_id=u1.id,
        referral_code="STATS-101",
        status="qualified",
        referrer_reward_awarded=True,
        reward_amount=5.00,
        fraud_flag=False,
    )
    ref2 = Referral(
        referrer_id=user.id,
        referred_id=u2.id,
        referral_code="STATS-101",
        status="pending",
        referrer_reward_awarded=False,
        reward_amount=5.00,
        fraud_flag=False,
    )
    db_session.add(ref1)
    db_session.add(ref2)
    db_session.commit()

    stats = await get_my_referrals(current_user=user, session=db_session, settings=settings)
    assert stats.referral_code == "STATS-101"
    assert stats.referral_link == "https://speedinfer.com/?ref=STATS-101"
    assert stats.total_referrals == 2
    assert stats.pending_referrals == 1
    assert stats.converted_referrals == 1
    assert stats.total_earnings_usd == 5.00
    assert stats.reward_per_referral_usd == 5.00
    assert stats.referee_bonus_usd == 5.00


# ---------------------------------------------------------------------------
# 8. Edge Case Tests: Key Provisioning on Verify & Profile Aliases
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_referee_bonus_provisions_key_when_registered_without_key(
    db_session: Session,
) -> None:
    """Ensure referee without an initial API key gets an active key funded with trial + bonus."""
    settings = Settings(
        api_key_pepper=SecretStr("test-pepper-32-chars-minimum-len"),
        trial_credit_balance=10.00,
        referee_bonus_amount=5.00,
    )

    referrer = User(email="referrer_nokey@speedinfer.com", referral_code="REF-NOKEY")
    db_session.add(referrer)
    db_session.commit()

    referee = User(email="referee_nokey@speedinfer.com", is_verified=False)
    db_session.add(referee)
    db_session.commit()
    db_session.refresh(referee)

    process_registration_referral(
        session=db_session,
        new_user=referee,
        referral_code="REF-NOKEY",
        device_fingerprint="fp-nokey-test",
        signup_ip_hash=hash_ip("198.51.100.1", "test-pepper"),
        settings=settings,
    )
    db_session.commit()

    # Verify no API key exists prior to email confirmation
    keys_before = db_session.exec(select(ApiKey).where(ApiKey.user_id == referee.id)).all()
    assert len(keys_before) == 0

    # Simulate email confirmation triggering award_referee_bonus
    awarded = award_referee_bonus(db_session, referee, settings=settings)
    db_session.commit()
    assert awarded is True

    # User now has an active key with trial ($10) + referee bonus ($5) = $15.00
    keys_after = db_session.exec(select(ApiKey).where(ApiKey.user_id == referee.id)).all()
    assert len(keys_after) == 1
    assert keys_after[0].is_active is True
    assert keys_after[0].credit_balance == 15.00


@pytest.mark.asyncio
async def test_profile_patch_with_aliases(db_session: Session) -> None:
    """Verify ProfileUpdateRequest accepts 'address' and 'company' as aliases."""
    from speedinfer.gateway.routes.auth import update_profile
    from speedinfer.gateway.schemas import ProfileUpdateRequest

    user = User(
        email="alias_profile_user@speedinfer.com",
        name="Old Name",
        location="Old Location",
        organization="Old Org",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Use 'address' instead of 'location', and 'company' instead of 'organization'
    req = ProfileUpdateRequest.model_validate(
        {
            "name": "New Name",
            "address": "San Francisco, CA",
            "company": "Acme Innovations",
        }
    )
    assert req.location == "San Francisco, CA"
    assert req.organization == "Acme Innovations"

    resp = await update_profile(payload=req, current_user=user, session=db_session)
    assert resp.name == "New Name"
    assert resp.location == "San Francisco, CA"
    assert resp.address == "San Francisco, CA"
    assert resp.organization == "Acme Innovations"
    assert resp.company == "Acme Innovations"
