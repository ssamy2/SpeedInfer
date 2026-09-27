"""SQLModel database models for the SpeedInfer platform persistence layer.

This module defines the persistent entities that serve as the single source
of truth across all SpeedInfer components:
- User: Identity and account management.
- ApiKey: Authentication credentials, rate limits, credit balance, and scopes.
- UsageLedger: Immutable auditing for inference transactions, token accounting, and costs.
- ModelVersion: Model registry, serving metadata, token pricing, and lifecycle states.
"""

import json
import re
import secrets
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import ConfigDict, field_validator
from sqlalchemy import CheckConstraint, Column, Index, LargeBinary, Text, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel

from speedinfer.database.workspace import (  # noqa: F401
    PolicyAcceptance,
    WorkspaceObject,
    WorkspaceResource,
)


def utc_now() -> datetime:
    """Return the current timestamp in UTC timezone.

    Used as the default factory across all model timestamp fields to ensure
    consistent, timezone-aware datetime representations without deprecation warnings.
    """
    return datetime.now(UTC)


def generate_referral_code() -> str:
    """Generate a secure, unambiguous 8-character referral code."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(8))


class LifecycleStatus(StrEnum):
    """Lifecycle statuses for model serving and registry management."""

    ACTIVE = "active"
    EVALUATING = "evaluating"
    DEPRECATED = "deprecated"
    TRAINING = "training"
    STAGING = "staging"
    FAILED_EVALUATION = "failed_evaluation"


class User(SQLModel, table=True):
    """Platform user entity representing an account holder.

    Owns issued API keys and maintains global administrative flags.
    """

    model_config = ConfigDict(validate_assignment=True)
    __tablename__ = "user"

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Unique internal user identifier.",
    )
    email: str = Field(
        unique=True,
        index=True,
        nullable=False,
        max_length=255,
        description="User's primary email address, unique across the platform.",
    )
    name: str | None = Field(
        default=None,
        max_length=255,
        nullable=True,
        description="User's display name or organizational identifier.",
    )
    is_active: bool = Field(
        default=True,
        nullable=False,
        description="Whether the user account is active and permitted to use the platform.",
    )
    is_admin: bool = Field(
        default=False,
        nullable=False,
        description="Flag indicating system administrative privileges.",
    )
    password_hash: str | None = Field(
        default=None,
        nullable=True,
        max_length=255,
        description="Hashed password for user authentication.",
    )
    referral_code: str | None = Field(
        default_factory=generate_referral_code,
        unique=True,
        index=True,
        max_length=32,
        nullable=True,
        description="Unique referral code for user invite links.",
    )
    referred_by_id: int | None = Field(
        default=None,
        foreign_key="user.id",
        nullable=True,
        index=True,
        description="Optional ID of referring user.",
    )
    signup_ip_hash: str | None = Field(
        default=None,
        max_length=64,
        nullable=True,
        index=True,
        description="Hashed signup IP for anti-fraud.",
    )
    device_fingerprint: str | None = Field(
        default=None,
        max_length=128,
        nullable=True,
        index=True,
        description="Client device fingerprint hash for anti-fraud.",
    )
    referral_reward_claimed: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={"server_default": "0"},
        description="Whether referral conversion bonus was credited.",
    )
    is_verified: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={"server_default": "0"},
        description="Whether user email has been verified.",
    )
    email_verified_at: datetime | None = Field(
        default=None,
        nullable=True,
        description="Timestamp of email verification.",
    )
    avatar_url: str | None = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Avatar image URL or base64 data URI.",
    )
    location: str | None = Field(
        default=None,
        max_length=255,
        nullable=True,
        description="User address or location.",
    )
    organization: str | None = Field(
        default=None,
        max_length=255,
        nullable=True,
        description="User organization or company name.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        description="Timestamp when user was registered in UTC.",
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        description="Timestamp when user was last updated in UTC.",
    )

    api_keys: list["ApiKey"] = Relationship(
        back_populates="user",
        cascade_delete=True,
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    oauth_accounts: list["OAuthAccount"] = Relationship(
        back_populates="user",
        cascade_delete=True,
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        """Validate and normalize email address format."""
        normalized = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", normalized):
            raise ValueError(f"Invalid email address format: '{value}'")
        return normalized

    def touch(self) -> None:
        """Update the updated_at timestamp to current UTC time."""
        self.updated_at = utc_now()


class ApiKey(SQLModel, table=True):
    """Bearer API credential entity for accessing SpeedInfer inference services.

    Stores cryptographic SHA-256 HMAC hash of the secret key, token bucket limits,
    prepaid credit balances, and permission scopes.
    """

    model_config = ConfigDict(validate_assignment=True)
    __tablename__ = "apikey"
    __table_args__ = (
        CheckConstraint("credit_balance >= 0.0", name="check_apikey_credit_balance_non_negative"),
        CheckConstraint("trial_balance >= 0.0", name="check_apikey_trial_balance_non_negative"),
        CheckConstraint("paid_balance >= 0.0", name="check_apikey_paid_balance_non_negative"),
        CheckConstraint("rpm_limit > 0", name="check_apikey_rpm_limit_positive"),
        CheckConstraint("tpm_limit > 0", name="check_apikey_tpm_limit_positive"),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Unique internal API key identifier.",
    )
    user_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="Foreign key to the owning User.",
    )
    key_hash: str = Field(
        unique=True,
        index=True,
        nullable=False,
        max_length=64,
        description="Hex-encoded SHA-256 HMAC hash of the plaintext secret key with pepper.",
    )
    prefix: str = Field(
        index=True,
        nullable=False,
        max_length=32,
        description="Publicly safe prefix for identification (e.g. sk-speedinfer-1234abcd).",
    )
    name: str = Field(
        default="default",
        max_length=100,
        nullable=False,
        description="Human-readable label for this API key.",
    )
    permissions: str = Field(
        default="chat:completions,completions,models:read,usage:read",
        max_length=512,
        nullable=False,
        description="Comma-separated or JSON list of permission scopes granted to this key.",
    )
    trial_balance: float = Field(
        default=0.0,
        nullable=False,
        sa_column_kwargs={"server_default": "0.0"},
        description=(
            "Current promotional, referral, or trial balance in USD available for inference calls."
        ),
    )
    paid_balance: float = Field(
        default=0.0,
        nullable=False,
        sa_column_kwargs={"server_default": "0.0"},
        description="Current real paid/topped-up balance in USD available for inference calls.",
    )
    credit_balance: float = Field(
        default=0.0,
        nullable=False,
        sa_column_kwargs={"server_default": "0.0"},
        description="Total available balance in USD (trial_balance + paid_balance).",
    )
    rpm_limit: int = Field(
        default=60,
        nullable=False,
        description="Requests-per-minute rate limit for this key.",
    )
    tpm_limit: int = Field(
        default=60_000,
        nullable=False,
        description="Tokens-per-minute rate limit for this key.",
    )
    is_active: bool = Field(
        default=True,
        nullable=False,
        description="Whether this API key is active. Inactive keys are rejected at auth gateway.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        description="Timestamp when API key was created in UTC.",
    )
    expires_at: datetime | None = Field(
        default=None,
        nullable=True,
        description="Optional expiration timestamp in UTC. Null represents non-expiring key.",
    )
    last_used_at: datetime | None = Field(
        default=None,
        nullable=True,
        description="Timestamp when key was last used for an inference call in UTC.",
    )

    user: User | None = Relationship(back_populates="api_keys")
    usage_records: list["UsageLedger"] = Relationship(
        back_populates="api_key",
        cascade_delete=True,
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )

    def __init__(self, **data: Any) -> None:
        """Initialize ApiKey instance with support for key_prefix alias and dual balance."""
        if "key_prefix" in data and "prefix" not in data:
            data["prefix"] = data.pop("key_prefix")
        trial = float(data.get("trial_balance", 0.0))
        paid = float(data.get("paid_balance", 0.0))
        if "credit_balance" in data and "trial_balance" not in data and "paid_balance" not in data:
            cb = float(data["credit_balance"])
            data["trial_balance"] = 0.0
            data["paid_balance"] = cb
            data["credit_balance"] = cb
        elif "credit_balance" not in data:
            data["credit_balance"] = round(trial + paid, 6)
        super().__init__(**data)

    @property
    def key_prefix(self) -> str:
        """Return the public key prefix as an alias to prefix."""
        return self.prefix

    @key_prefix.setter
    def key_prefix(self, value: str) -> None:
        """Set the public key prefix via key_prefix alias."""
        self.prefix = value

    @field_validator("key_hash")
    @classmethod
    def validate_key_hash(cls, value: str) -> str:
        """Ensure key_hash is a 64-character lowercase hex string."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or not all(c in "0123456789abcdef" for c in normalized):
            raise ValueError("key_hash must be a 64-character lowercase hex string")
        return normalized

    @field_validator("prefix")
    @classmethod
    def validate_prefix(cls, value: str) -> str:
        """Ensure key prefix is non-empty and stripped."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("prefix cannot be empty")
        return normalized

    @field_validator("permissions", mode="before")
    @classmethod
    def validate_permissions(cls, value: str | list[str]) -> str:
        """Normalize permissions list or string to a canonical comma-separated string."""
        if isinstance(value, list):
            return ",".join(str(p).strip() for p in value if str(p).strip())
        if isinstance(value, str):
            return ",".join(str(p).strip() for p in value.split(",") if str(p).strip())
        raise ValueError("permissions must be a comma-separated string or a list of strings")

    @field_validator("trial_balance", "paid_balance", "credit_balance")
    @classmethod
    def validate_credit_balance(cls, value: float) -> float:
        """Ensure credit balance is non-negative and rounded to 6 decimal places."""
        if value < 0.0:
            raise ValueError("credit_balance cannot be negative")
        return round(float(value), 6)

    @field_validator("rpm_limit", "tpm_limit")
    @classmethod
    def validate_positive_limit(cls, value: int) -> int:
        """Ensure rate limits are strictly positive integers."""
        if value <= 0:
            raise ValueError("rate limits must be strictly positive integers")
        return value

    def get_permissions(self) -> list[str]:
        """Parse permissions attribute into a list of normalized scope strings."""
        raw = self.permissions.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(p).strip() for p in parsed if str(p).strip()]
            except json.JSONDecodeError:
                pass
        return [p.strip() for p in raw.split(",") if p.strip()]

    def has_permission(self, permission: str) -> bool:
        """Check if this API key grants the specified permission scope.

        The 'admin' scope acts as a wildcard granting access to all operations.
        """
        scopes = self.get_permissions()
        return "admin" in scopes or permission in scopes

    def is_expired(self) -> bool:
        """Check whether this API key has passed its expiration timestamp."""
        if self.expires_at is None:
            return False
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        return expires < utc_now()

    def is_valid(self) -> bool:
        """Check whether this API key is active and not expired."""
        return self.is_active and not self.is_expired()


class UsageLedger(SQLModel, table=True):
    """Immutable audit ledger recording each completed inference transaction.

    Captures token accounting, latencies, billing cost, and HTTP status codes.
    """

    model_config = ConfigDict(validate_assignment=True)
    __tablename__ = "usageledger"
    __table_args__ = (
        Index("ix_usageledger_api_key_created_at", "api_key_id", "created_at"),
        CheckConstraint("prompt_tokens >= 0", name="check_usageledger_prompt_tokens_non_negative"),
        CheckConstraint(
            "completion_tokens >= 0", name="check_usageledger_completion_tokens_non_negative"
        ),
        CheckConstraint("total_tokens >= 0", name="check_usageledger_total_tokens_non_negative"),
        CheckConstraint("total_cost >= 0.0", name="check_usageledger_total_cost_non_negative"),
        CheckConstraint("latency_ms >= 0.0", name="check_usageledger_latency_ms_non_negative"),
        CheckConstraint(
            "status_code >= 100 AND status_code <= 599",
            name="check_usageledger_status_code_valid",
        ),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Unique internal ledger record identifier.",
    )
    api_key_id: int = Field(
        foreign_key="apikey.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="Foreign key to the ApiKey billed for this transaction.",
    )
    request_id: str = Field(
        unique=True,
        index=True,
        nullable=False,
        max_length=64,
        description="Unique correlated request ID.",
    )
    model: str = Field(
        index=True,
        nullable=False,
        max_length=128,
        description="Name of the model that executed the inference.",
    )
    prompt_tokens: int = Field(
        default=0,
        nullable=False,
        description="Number of prompt tokens evaluated in the request.",
    )
    completion_tokens: int = Field(
        default=0,
        nullable=False,
        description="Number of completion tokens generated in the response.",
    )
    total_tokens: int = Field(
        default=0,
        nullable=False,
        description="Total tokens (prompt + completion) consumed in the transaction.",
    )
    total_cost: float = Field(
        default=0.0,
        nullable=False,
        description="Total USD cost calculated and deducted for this request.",
    )
    latency_ms: float = Field(
        default=0.0,
        nullable=False,
        description="End-to-end request duration in milliseconds.",
    )
    ttft_ms: float | None = Field(
        default=None,
        nullable=True,
        description="Time to first token in milliseconds for streaming responses.",
    )
    status_code: int = Field(
        default=200,
        nullable=False,
        description="HTTP status code returned to the client.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        index=True,
        description="Timestamp when the inference transaction concluded in UTC.",
    )

    api_key: ApiKey | None = Relationship(back_populates="usage_records")

    def __init__(self, **data: Any) -> None:
        """Initialize UsageLedger with cost alias and total_tokens auto-derivation."""
        if "cost" in data and "total_cost" not in data:
            data["total_cost"] = data.pop("cost")
        if "total_tokens" not in data and "prompt_tokens" in data and "completion_tokens" in data:
            data["total_tokens"] = data["prompt_tokens"] + data["completion_tokens"]
        super().__init__(**data)

    @property
    def cost(self) -> float:
        """Return total_cost as an alias to cost."""
        return self.total_cost

    @cost.setter
    def cost(self, value: float) -> None:
        """Set total_cost via cost alias."""
        self.total_cost = value

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens")
    @classmethod
    def validate_non_negative_tokens(cls, value: int) -> int:
        """Ensure token counts are non-negative."""
        if value < 0:
            raise ValueError("Token counts cannot be negative")
        return value

    @field_validator("total_cost", "latency_ms")
    @classmethod
    def validate_non_negative_floats(cls, value: float) -> float:
        """Ensure cost and latency are non-negative."""
        if value < 0.0:
            raise ValueError("Cost and latency values cannot be negative")
        return round(float(value), 6)

    @field_validator("ttft_ms")
    @classmethod
    def validate_ttft(cls, value: float | None) -> float | None:
        """Ensure TTFT is non-negative when provided."""
        if value is not None and value < 0.0:
            raise ValueError("Time to first token (ttft_ms) cannot be negative")
        return round(float(value), 3) if value is not None else None

    @field_validator("status_code")
    @classmethod
    def validate_status_code(cls, value: int) -> int:
        """Ensure status code is a valid HTTP status code."""
        if value < 100 or value > 599:
            raise ValueError("status_code must be a valid HTTP code between 100 and 599")
        return value


class PaymentTransaction(SQLModel, table=True):
    """Idempotent record of a successful external payment credited to an API key."""

    __tablename__ = "paymenttransaction"
    __table_args__ = (
        CheckConstraint("amount_usd > 0.0", name="check_paymenttransaction_amount_positive"),
        CheckConstraint("credits_added > 0.0", name="check_paymenttransaction_credits_positive"),
    )

    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(default="whop", nullable=False, max_length=32)
    provider_payment_id: str = Field(unique=True, index=True, nullable=False, max_length=128)
    webhook_id: str = Field(unique=True, index=True, nullable=False, max_length=128)
    user_id: int = Field(foreign_key="user.id", index=True, nullable=False, ondelete="CASCADE")
    api_key_id: int = Field(foreign_key="apikey.id", index=True, nullable=False, ondelete="CASCADE")
    amount_usd: float = Field(nullable=False)
    credits_added: float = Field(nullable=False)
    currency: str = Field(default="usd", nullable=False, max_length=8)
    created_at: datetime = Field(default_factory=utc_now, nullable=False, index=True)


class ModelVersion(SQLModel, table=True):
    """Model version and serving registry entity.

    Tracks available model architectures, weight locations, LoRA adapters,
    token pricing rates, and serving lifecycle status.
    """

    model_config = ConfigDict(validate_assignment=True)
    __tablename__ = "modelversion"
    __table_args__ = (
        CheckConstraint("context_length > 0", name="check_modelversion_context_length_positive"),
        CheckConstraint(
            "prompt_price_per_million >= 0.0",
            name="check_modelversion_prompt_price_non_negative",
        ),
        CheckConstraint(
            "completion_price_per_million >= 0.0",
            name="check_modelversion_completion_price_non_negative",
        ),
        CheckConstraint(
            "lifecycle_status IN ('active', 'evaluating', 'deprecated', "
            "'training', 'staging', 'failed_evaluation')",
            name="check_modelversion_lifecycle_status_valid",
        ),
    )

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Unique internal model version identifier.",
    )
    name: str = Field(
        unique=True,
        index=True,
        nullable=False,
        max_length=128,
        description="Public model name exposed to clients via OpenAI API endpoints.",
    )
    base_model_path: str = Field(
        nullable=False,
        max_length=255,
        description="Filesystem path or HuggingFace repo ID for the base model weights.",
    )
    adapter_path: str | None = Field(
        default=None,
        nullable=True,
        max_length=512,
        description="Optional filesystem path to fine-tuned LoRA adapter weights.",
    )
    context_length: int = Field(
        default=32768,
        nullable=False,
        description="Maximum sequence context length supported by the model in tokens.",
    )
    prompt_price_per_million: float = Field(
        default=0.20,
        nullable=False,
        description="Price in USD per million prompt tokens evaluated.",
    )
    completion_price_per_million: float = Field(
        default=0.60,
        nullable=False,
        description="Price in USD per million completion tokens generated.",
    )
    lifecycle_status: str = Field(
        default=LifecycleStatus.ACTIVE.value,
        nullable=False,
        max_length=32,
        description="Current serving status: 'active', 'evaluating', or 'deprecated'.",
    )
    mlflow_run_id: str | None = Field(
        default=None,
        nullable=True,
        max_length=64,
        description="Correlated MLflow run ID if trained via speedinfer.training.",
    )
    mlflow_model_version: str | None = Field(
        default=None,
        nullable=True,
        max_length=32,
        description="MLflow model registry version tag.",
    )
    eval_metrics: str | None = Field(
        default=None,
        nullable=True,
        description="JSON-encoded evaluation metrics (e.g. perplexity, benchmark accuracy).",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        description="Timestamp when model version was registered in UTC.",
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        description="Timestamp when model version was last updated in UTC.",
    )

    def __init__(self, **data: Any) -> None:
        """Initialize ModelVersion with base_model and context_window aliases."""
        if "base_model" in data and "base_model_path" not in data:
            data["base_model_path"] = data.pop("base_model")
        if "context_window" in data and "context_length" not in data:
            data["context_length"] = data.pop("context_window")
        super().__init__(**data)

    @property
    def base_model(self) -> str:
        """Return base_model_path via base_model alias."""
        return self.base_model_path

    @base_model.setter
    def base_model(self, value: str) -> None:
        """Set base_model_path via base_model alias."""
        self.base_model_path = value

    @property
    def context_window(self) -> int:
        """Return context_length via context_window alias."""
        return self.context_length

    @context_window.setter
    def context_window(self, value: int) -> None:
        """Set context_length via context_window alias."""
        self.context_length = value

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, value: int) -> int:
        """Ensure context length is strictly positive."""
        if value <= 0:
            raise ValueError("context_length must be strictly positive")
        return value

    @field_validator("prompt_price_per_million", "completion_price_per_million")
    @classmethod
    def validate_pricing(cls, value: float) -> float:
        """Ensure token prices are non-negative."""
        if value < 0.0:
            raise ValueError("Token prices cannot be negative")
        return round(float(value), 6)

    @field_validator("lifecycle_status")
    @classmethod
    def validate_lifecycle_status(cls, value: str | LifecycleStatus) -> str:
        """Validate that lifecycle_status belongs to the allowed LifecycleStatus enum set."""
        status_str = (
            value.value if isinstance(value, LifecycleStatus) else str(value).strip().lower()
        )
        status_map = {
            "production": LifecycleStatus.ACTIVE.value,
            "staged": LifecycleStatus.STAGING.value,
        }
        status_str = status_map.get(status_str, status_str)
        allowed = {s.value for s in LifecycleStatus}
        if status_str not in allowed:
            raise ValueError(f"Invalid lifecycle_status: '{value}'. Allowed: {sorted(allowed)}")
        return status_str

    def is_available(self) -> bool:
        """Return True if the model version is actively serving inference traffic."""
        return self.lifecycle_status == LifecycleStatus.ACTIVE.value

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the total billing cost for a given token usage count."""
        prompt_cost = (prompt_tokens / 1_000_000.0) * self.prompt_price_per_million
        completion_cost = (completion_tokens / 1_000_000.0) * self.completion_price_per_million
        return round(prompt_cost + completion_cost, 6)

    def touch(self) -> None:
        """Update the updated_at timestamp to current UTC time."""
        self.updated_at = utc_now()


class ContactRequest(SQLModel, table=True):
    """Private sales/support inbox. Never expose request contents publicly."""

    __tablename__ = "contactrequest"

    id: str = Field(primary_key=True, max_length=36)
    category: str = Field(max_length=16)
    name: str = Field(max_length=100)
    email: str = Field(max_length=254, index=True)
    company: str = Field(default="", max_length=120)
    subject: str = Field(max_length=160)
    message: str = Field(max_length=3000)
    client_fingerprint: str = Field(max_length=64, index=True)
    delivery_status: str = Field(default="pending", max_length=16)
    created_at: datetime = Field(default_factory=utc_now, index=True)


class TrialCreditGrant(SQLModel, table=True):
    """One durable entitlement marker per account, surviving key revocation/deletion."""

    __tablename__ = "trial_credit_grant"
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    amount: float
    created_at: datetime = Field(default_factory=utc_now)


class OAuthAccount(SQLModel, table=True):
    """Linked external OAuth credentials (Google, GitHub) for social login."""

    model_config = ConfigDict(validate_assignment=True)
    __tablename__ = "oauth_account"
    __table_args__ = (Index("ix_oauth_provider_uid", "provider", "provider_user_id", unique=True),)

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Internal OAuth link identifier.",
    )
    user_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="ID of the SpeedInfer user account.",
    )
    provider: str = Field(
        index=True,
        nullable=False,
        max_length=32,
        description="Identity provider name (e.g., 'google', 'github').",
    )
    provider_user_id: str = Field(
        index=True,
        nullable=False,
        max_length=128,
        description="Unique user ID from identity provider.",
    )
    provider_email: str | None = Field(
        default=None,
        max_length=255,
        nullable=True,
        description="Email reported by identity provider.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        description="Timestamp when OAuth link was established in UTC.",
    )

    user: User | None = Relationship(back_populates="oauth_accounts")


class EmailVerificationCode(SQLModel, table=True):
    """Time-limited 6-digit OTP codes for email verification and password resets."""

    model_config = ConfigDict(validate_assignment=True)
    __tablename__ = "email_verification_code"
    __table_args__ = (Index("ix_email_code_purpose", "email", "code", "purpose"),)

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Unique code record identifier.",
    )
    email: str = Field(
        index=True,
        nullable=False,
        max_length=255,
        description="Target email address.",
    )
    code: str = Field(
        index=True,
        nullable=False,
        max_length=8,
        description="6-digit verification OTP code.",
    )
    purpose: str = Field(
        default="registration",
        max_length=32,
        description="Purpose of OTP code: 'registration' or 'password_reset'.",
    )
    attempts: int = Field(
        default=0,
        nullable=False,
        description="Number of failed verification attempts.",
    )
    is_used: bool = Field(
        default=False,
        nullable=False,
        description="Whether this code has been successfully consumed.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        description="Timestamp when code was issued.",
    )
    expires_at: datetime = Field(
        nullable=False,
        description="Timestamp when code expires in UTC.",
    )


class Referral(SQLModel, table=True):
    """Referral attribution, anti-fraud evaluation, and reward tracking."""

    model_config = ConfigDict(validate_assignment=True)
    __tablename__ = "referral"

    id: int | None = Field(
        default=None,
        primary_key=True,
        description="Internal referral record identifier.",
    )
    referrer_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="User ID of the inviter.",
    )
    referred_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        unique=True,
        ondelete="CASCADE",
        description="User ID of the invited referee.",
    )
    status: str = Field(
        default="pending",
        max_length=32,
        description="Status: 'pending', 'rewarded', or 'rejected_fraud'.",
    )
    referee_bonus_awarded: bool = Field(
        default=False,
        nullable=False,
        description="Whether referee received the signup bonus (+ $5.00).",
    )
    referrer_reward_awarded: bool = Field(
        default=False,
        nullable=False,
        description="Whether referrer received the conversion reward (+ $5.00).",
    )
    reward_amount: float = Field(
        default=5.0,
        nullable=False,
        description="USD bonus amount for referrer upon qualification.",
    )
    device_fingerprint: str | None = Field(
        default=None,
        max_length=128,
        nullable=True,
        index=True,
        description="Client device fingerprint hash of referee.",
    )
    signup_ip_hash: str | None = Field(
        default=None,
        max_length=64,
        nullable=True,
        index=True,
        description="Hashed client IP of referee.",
    )
    fraud_flag: bool = Field(
        default=False,
        nullable=False,
        description="True if flagged by anti-fraud heuristics.",
    )
    fraud_reason: str | None = Field(
        default=None,
        max_length=255,
        nullable=True,
        description="Reason for fraud classification if flagged.",
    )
    created_at: datetime = Field(
        default_factory=utc_now,
        nullable=False,
        index=True,
        description="Timestamp when referral was registered.",
    )
    rewarded_at: datetime | None = Field(
        default=None,
        nullable=True,
        description="Timestamp when referrer bonus was granted.",
    )


class FileRecord(SQLModel, table=True):
    """Developer API uploaded file entity (OpenAI compatible)."""

    __tablename__ = "file_record"

    id: str = Field(
        default_factory=lambda: f"file-{secrets.token_hex(12)}",
        primary_key=True,
        description="Unique file identifier (file-...).",
    )
    user_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="Owner user ID.",
    )
    filename: str = Field(max_length=255, nullable=False, description="Original filename.")
    size_bytes: int = Field(default=0, nullable=False, description="File size in bytes.")
    purpose: str = Field(default="fine-tune", max_length=64, index=True, nullable=False)
    status: str = Field(default="uploaded", max_length=32, nullable=False)
    content: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
        nullable=False,
        index=True,
    )


class BucketRecord(SQLModel, table=True):
    """Developer API storage bucket entity."""

    __tablename__ = "bucket_record"
    __table_args__ = (UniqueConstraint("user_id", "name"),)

    id: str = Field(
        default_factory=lambda: f"bkt-{secrets.token_hex(8)}",
        primary_key=True,
        description="Unique bucket identifier.",
    )
    user_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="Owner user ID.",
    )
    name: str = Field(max_length=128, index=True, nullable=False)
    description: str | None = Field(default=None, max_length=512, nullable=True)
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class BucketObjectRecord(SQLModel, table=True):
    """Developer API storage object within a bucket."""

    __tablename__ = "bucket_object_record"
    __table_args__ = (UniqueConstraint("bucket_id", "name"),)

    id: str = Field(
        default_factory=lambda: f"obj-{secrets.token_hex(8)}",
        primary_key=True,
        description="Unique object identifier.",
    )
    user_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="Owner user ID.",
    )
    bucket_id: str = Field(
        foreign_key="bucket_record.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
    )
    name: str = Field(max_length=255, index=True, nullable=False)
    size: int = Field(default=0, nullable=False)
    content_type: str = Field(default="application/octet-stream", max_length=128)
    content: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    created_at: datetime = Field(default_factory=utc_now, nullable=False)


class FineTuningJobRecord(SQLModel, table=True):
    """Developer API model fine-tuning job entity."""

    __tablename__ = "fine_tuning_job_record"

    id: str = Field(
        default_factory=lambda: f"ftjob-{secrets.token_hex(12)}",
        primary_key=True,
        description="Unique fine-tuning job identifier.",
    )
    user_id: int = Field(
        foreign_key="user.id",
        index=True,
        nullable=False,
        ondelete="CASCADE",
        description="Owner user ID.",
    )
    model: str = Field(max_length=200, nullable=False, description="Base model identifier.")
    training_file_id: str = Field(max_length=64, nullable=False)
    validation_file_id: str | None = Field(default=None, max_length=64, nullable=True)
    status: str = Field(default="queued", max_length=32, index=True, nullable=False)
    fine_tuned_model: str | None = Field(default=None, max_length=255, nullable=True)
    hyperparameters_json: str = Field(default="{}", sa_column=Column(Text, nullable=False))
    trained_tokens: int = Field(default=0, nullable=False)
    checkpoints_json: str = Field(default="[]", sa_column=Column(Text, nullable=False))
    events_json: str = Field(default="[]", sa_column=Column(Text, nullable=False))
    error_json: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    created_at: int = Field(
        default_factory=lambda: int(datetime.now(UTC).timestamp()),
        nullable=False,
        index=True,
    )
    finished_at: int | None = Field(default=None, nullable=True)
