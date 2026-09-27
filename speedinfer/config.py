"""Validated application settings loaded from environment variables."""

import re
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the gateway and its persistence layer."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "SpeedInfer"
    environment: str = "development"
    log_level: str = "INFO"
    database_url: str = "sqlite:///./data/speedinfer.db"
    redis_url: str = "redis://localhost:6379/0"
    api_key_pepper: SecretStr = Field(min_length=16)
    default_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_base_url: str = "http://vllm:8000"
    vllm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    prompt_price_per_million: float = Field(default=0.20, ge=0)
    completion_price_per_million: float = Field(default=0.60, ge=0)
    max_request_tokens: int = Field(default=32768, gt=0, le=1_000_000)
    public_base_url: str = "http://localhost:8000"
    jwt_secret_key: SecretStr | None = None
    # Contact requests are always stored; SMTP optionally notifies the team inbox.
    sales_email: str = "Sales@speedinfer.com"
    support_email: str = "Support@speedinfer.com"
    smtp_host: str = ""
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: str = ""
    smtp_password: SecretStr | None = None
    smtp_from_email: str = ""
    smtp_implicit_tls: bool = False
    smtp_timeout_seconds: float = Field(default=10, gt=0, le=30)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = Field(default=1440, gt=0)
    # Billing is intentionally server-side. Never expose either value to the browser.
    whop_api_key: SecretStr | None = None
    whop_webhook_secret: SecretStr | None = None
    whop_account_id: str = ""
    whop_api_version_date: str = "2026-09-25"
    whop_credit_packages: str = "10,25,50,100"
    trial_credit_balance: float = Field(default=0.0, ge=0.0, le=1000.0)
    # OAuth Providers
    github_client_id: str = ""
    github_client_secret: SecretStr | None = None
    github_redirect_uri: str = "https://speedinfer.com/v1/auth/oauth/github/callback"
    google_client_id: str = ""
    google_client_secret: SecretStr | None = None
    google_redirect_uri: str = "https://speedinfer.com/v1/auth/oauth/google/callback"
    # Cloudflare Turnstile
    turnstile_site_key: str = "0x4AAAAAAFFIh2opZ2El7AU9"
    turnstile_secret_key: SecretStr | None = None
    turnstile_enabled: bool = False
    turnstile_hostnames: str = "speedinfer.com,localhost,127.0.0.1"
    # Referral Program
    referral_reward_amount: float = Field(default=2.0, ge=0.0)
    referee_bonus_amount: float = Field(default=0.0, ge=0.0)
    referral_min_topup: float = Field(default=20.0, ge=0.0)

    @field_validator("sales_email", "support_email", "smtp_from_email")
    @classmethod
    def validate_contact_email(cls, value: str, info) -> str:
        value = value.strip()
        if not value and info.field_name == "smtp_from_email":
            return value
        if not re.fullmatch(
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}", value
        ):
            raise ValueError("Contact email must be a single valid mailbox address.")
        return value

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, value: str) -> str:
        """Reject accidental empty deployment environment names."""
        normalized = value.strip().lower()
        if normalized not in {"development", "test", "staging", "production"}:
            raise ValueError("environment must be development, test, staging, or production")
        return normalized

    @field_validator("jwt_algorithm")
    @classmethod
    def validate_jwt_algorithm(cls, value: str) -> str:
        """Ensure JWT signing algorithm is an approved HMAC variant."""
        normalized = value.strip().upper()
        if normalized not in {"HS256", "HS384", "HS512"}:
            raise ValueError(f"jwt_algorithm must be one of HS256, HS384, HS512; got '{value}'")
        return normalized

    def credit_packages(self) -> tuple[float, ...]:
        """Return the configured, safe-to-sell prepaid USD credit packages."""
        try:
            package_values = self.whop_credit_packages.split(",")
            values = tuple(
                sorted({round(float(item.strip()), 2) for item in package_values})
            )
        except ValueError as exc:
            raise ValueError("WHOP_CREDIT_PACKAGES must be comma-separated USD amounts.") from exc
        if not values or any(value <= 0 or value > 10_000 for value in values):
            raise ValueError("WHOP_CREDIT_PACKAGES must contain values between 0 and 10000.")
        return values


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable-by-convention settings object."""
    return Settings()
