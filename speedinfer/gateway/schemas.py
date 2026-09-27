"""OpenAI-compatible Pydantic request and response schemas.

Defines schemas for:
- /v1/chat/completions (streaming and non-streaming)
- /v1/completions (legacy text completions)
- /v1/models catalog inspection
- /v1/usage credit balance inspection
- /health diagnostics
- RFC 7807 / OpenAI error envelope
"""

import re
import time
from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class ChatMessage(BaseModel):
    """Individual chat completion message."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message author (system, user, assistant)."
    )
    content: str = Field(max_length=1_000_000, description="Textual content of the message.")
    name: str | None = Field(default=None, description="Optional name of the participant.")


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = True


class ChatCompletionRequest(BaseModel):
    """Payload for POST /v1/chat/completions."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str = Field(description="Target model identifier.")
    stream_options: StreamOptions | None = None
    messages: list[ChatMessage] = Field(description="Sequential list of conversation messages.")
    temperature: float | None = Field(
        default=1.0, description="Sampling temperature between 0.0 and 2.0."
    )
    top_p: float | None = Field(
        default=1.0, gt=0, le=1, description="Nucleus sampling probability."
    )
    n: int | None = Field(default=1, ge=1, le=1, description="Number of completions to generate.")
    stream: bool | None = Field(
        default=False, description="Whether to stream back partial progress via SSE."
    )
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequence(s) to abort generation."
    )
    max_tokens: int | None = Field(
        default=128, gt=0, le=32768, description="Maximum number of tokens to generate."
    )
    presence_penalty: float | None = Field(
        default=0.0, description="Presence penalty between -2.0 and 2.0."
    )
    frequency_penalty: float | None = Field(
        default=0.0, description="Frequency penalty between -2.0 and 2.0."
    )
    user: str | None = Field(default=None, description="Unique identifier representing end-user.")

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        """Ensure messages array contains at least one message."""
        if not value:
            raise ValueError("messages array cannot be empty")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature_range(cls, value: float | None) -> float | None:
        """Validate temperature is within OpenAI bounds [0.0, 2.0]."""
        if value is not None and (value < 0.0 or value > 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value


class CompletionRequest(BaseModel):
    """Payload for POST /v1/completions legacy endpoint."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    model: str = Field(description="Target model identifier.")
    prompt: str | list[str] = Field(description="The prompt(s) to generate completions for.")
    max_tokens: int | None = Field(
        default=16, gt=0, le=32768, description="Maximum number of tokens to generate."
    )
    temperature: float | None = Field(
        default=1.0, description="Sampling temperature between 0.0 and 2.0."
    )
    top_p: float | None = Field(
        default=1.0, gt=0, le=1, description="Nucleus sampling probability."
    )
    n: int | None = Field(default=1, ge=1, le=1, description="Number of completions to generate.")
    stream: bool | None = Field(
        default=False, description="Whether to stream back partial progress via SSE."
    )
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequence(s) to abort generation."
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt_non_empty(cls, value: str | list[str]) -> str | list[str]:
        """Ensure prompt is not an empty string or empty array."""
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("prompt cannot be an empty string")
        elif isinstance(value, list):
            if not value:
                raise ValueError("prompt array cannot be empty")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature_range(cls, value: float | None) -> float | None:
        """Validate temperature is within [0.0, 2.0]."""
        if value is not None and (value < 0.0 or value > 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return value


class UsageInfo(BaseModel):
    """Token consumption accounting metrics."""

    prompt_tokens: int = Field(ge=0, description="Tokens evaluated in prompt context.")
    completion_tokens: int = Field(ge=0, description="Tokens produced in generation.")
    total_tokens: int = Field(ge=0, description="Sum of prompt and completion tokens.")

    @model_validator(mode="after")
    def consistent_total(self):
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("Worker usage total is inconsistent")
        return self


class ChatResponseMessage(BaseModel):
    """Assistant message payload in completion choices."""

    role: str = Field(default="assistant", description="Role of the author.")
    content: str = Field(description="Generated textual content.")


class ChatCompletionChoice(BaseModel):
    """Single choice item in chat completion response."""

    index: int = Field(default=0, description="Choice index.")
    message: ChatResponseMessage = Field(description="Message generated by assistant.")
    finish_reason: Literal["stop", "length", "content_filter"] = Field(
        default="stop", description="Reason token generation terminated."
    )


class ChatCompletionResponse(BaseModel):
    """Standard non-streaming OpenAI chat completion response."""

    id: str = Field(description="Unique completion identifier prefix with chatcmpl-.")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()), description="Unix timestamp.")
    model: str = Field(description="Model identifier that produced the completion.")
    choices: list[ChatCompletionChoice] = Field(description="Generated candidate completions.")
    usage: UsageInfo = Field(description="Token accounting breakdown.")


class DeltaMessage(BaseModel):
    """Delta payload in streaming chunk."""

    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice item in streaming chunk."""

    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """Server-Sent Event streaming chunk payload."""

    id: str = Field(description="Unique completion chunk identifier.")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(description="Model identifier.")
    choices: list[ChatCompletionChunkChoice] = Field(description="Chunk token deltas.")


class CompletionChoice(BaseModel):
    """Choice item in legacy text completion response."""

    text: str = Field(description="Generated continuation text.")
    index: int = Field(default=0, description="Candidate index.")
    logprobs: Any | None = Field(default=None, description="Log probabilities if requested.")
    finish_reason: Literal["stop", "length"] = Field(
        default="stop", description="Termination reason."
    )


class CompletionResponse(BaseModel):
    """Standard OpenAI legacy text completion response."""

    id: str = Field(description="Unique completion identifier prefixed with cmpl-.")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = Field(description="Model identifier.")
    choices: list[CompletionChoice] = Field(description="Generated continuations.")
    usage: UsageInfo = Field(description="Token accounting breakdown.")


class ModelObject(BaseModel):
    """OpenAI model representation."""

    id: str = Field(description="Model identifier.")
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = Field(default="speedinfer")
    permission: list[Any] = Field(default_factory=list)
    root: str | None = None
    parent: str | None = None
    context_length: int | None = None
    prompt_price_per_million: float | None = None
    completion_price_per_million: float | None = None


class ModelListResponse(BaseModel):
    """OpenAI GET /v1/models response catalog."""

    object: Literal["list"] = "list"
    data: list[ModelObject] = Field(description="Available active models.")


class UsageResponse(BaseModel):
    """Current user usage and credit balance response."""

    credit_balance: float = Field(description="Current available credit in USD.")
    balance: float = Field(description="Alias for credit_balance.")
    paid_balance: float = Field(default=0.0, description="Real paid balance in USD.")
    trial_balance: float = Field(default=0.0, description="Promotional/trial balance in USD.")
    currency: str = Field(default="USD", description="Currency denomination.")
    rpm_limit: int = Field(description="Requests per minute rate limit.")
    tpm_limit: int = Field(description="Tokens per minute rate limit.")


class HealthResponse(BaseModel):
    """Diagnostic health check response."""

    status: str = Field(default="healthy")
    service: str = Field(default="SpeedInfer")
    version: str = Field(default="0.1.0")
    timestamp: str = Field(description="ISO-8601 formatted timestamp.")
    models: list[str] = Field(default_factory=list)
    workers: list[dict[str, Any]] = Field(default_factory=list)
    gpu: dict[str, Any] = Field(default_factory=dict)


class UserRegisterRequest(BaseModel):
    """Payload for POST /v1/auth/register."""

    model_config = ConfigDict(extra="ignore")
    accepted_policy_version: Literal["2026-09-27", "2026-09-27-r2"] | None = None

    email: str = Field(description="User primary email address.")
    password: str = Field(
        min_length=6, max_length=256, description="User password (minimum 6 characters)."
    )
    name: str | None = Field(
        default=None, max_length=255, description="Optional user display name."
    )
    initial_balance: float = Field(
        default=10.0, ge=0.0, description="Initial trial balance in USD."
    )
    create_api_key: bool = Field(
        default=False, description="Explicit opt-in to create a default API key."
    )
    referral_code: str | None = Field(
        default=None, max_length=32, description="Optional referral code of inviter."
    )
    device_fingerprint: str | None = Field(
        default=None, max_length=128, description="Client device fingerprint hash for anti-fraud."
    )
    turnstile_token: str | None = Field(
        default=None, description="Cloudflare Turnstile verification token."
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        """Validate and normalize email format."""
        normalized = value.strip().lower()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", normalized):
            raise ValueError(f"Invalid email address format: '{value}'")
        return normalized

    @field_validator("referral_code")
    @classmethod
    def validate_referral_code_format(cls, value: str | None) -> str | None:
        """Normalize referral code to uppercase string."""
        if value is None:
            return None
        clean = value.strip().upper()
        return clean if clean else None

    @field_validator("device_fingerprint")
    @classmethod
    def validate_device_fp_format(cls, value: str | None) -> str | None:
        """Clean whitespace from device fingerprint."""
        if value is None:
            return None
        clean = value.strip()
        return clean if clean else None


class UserLoginRequest(BaseModel):
    """Payload for POST /v1/auth/login."""

    model_config = ConfigDict(extra="ignore")

    email: str = Field(description="User email address.")
    password: str = Field(max_length=256, description="User password.")
    turnstile_token: str | None = Field(
        default=None, description="Cloudflare Turnstile verification token."
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        """Normalize email address to lowercase."""
        return value.strip().lower()


class UserResponse(BaseModel):
    """User account details response."""

    id: int = Field(description="Unique internal user identifier.")
    email: str = Field(description="User email address.")
    name: str | None = Field(default=None, description="User display name.")
    is_active: bool = Field(default=True, description="Account active status.")
    is_admin: bool = Field(default=False, description="Admin status flag.")
    is_verified: bool = Field(default=False, description="Whether email has been verified.")
    referral_code: str | None = Field(
        default=None, description="User's unique referral invite code."
    )
    avatar_url: str | None = Field(default=None, description="User avatar image URL.")
    location: str | None = Field(default=None, description="User location or physical address.")
    address: str | None = Field(default=None, description="Alias for location.")
    organization: str | None = Field(default=None, description="User company or organization.")
    company: str | None = Field(default=None, description="Alias for organization.")
    created_at: datetime = Field(description="Account creation timestamp in UTC.")
    balance: float = Field(default=0.0, description="Available credit balance across keys in USD.")
    credit_balance: float = Field(default=0.0, description="Alias for balance in USD.")
    paid_balance: float = Field(default=0.0, description="Real paid balance across keys in USD.")
    trial_balance: float = Field(
        default=0.0, description="Promotional/trial balance across keys in USD."
    )


class VerifyEmailRequest(BaseModel):
    """Payload for POST /v1/auth/verify-email."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    email: str = Field(description="User email address.")
    code: str = Field(min_length=6, max_length=6, description="6-digit verification OTP code.")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("code")
    @classmethod
    def validate_code_format(cls, value: str) -> str:
        clean = value.strip()
        if not clean.isdigit() or len(clean) != 6:
            raise ValueError("Verification code must be 6 digits.")
        return clean


class ResendCodeRequest(BaseModel):
    """Payload for POST /v1/auth/resend-code."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    email: str = Field(description="User email address.")
    purpose: Literal["registration", "password_reset"] = Field(
        default="registration", description="Purpose of verification code."
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        return value.strip().lower()


class ForgotPasswordRequest(BaseModel):
    """Payload for POST /v1/auth/forgot-password."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    email: str = Field(description="Registered account email address.")
    turnstile_token: str | None = Field(default=None, description="Turnstile verification token.")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        return value.strip().lower()


class ResetPasswordRequest(BaseModel):
    """Payload for POST /v1/auth/reset-password."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    email: str = Field(description="Target account email address.")
    code: str = Field(min_length=6, max_length=6, description="6-digit password reset OTP.")
    new_password: str = Field(
        min_length=8, max_length=256, description="New password (min 8 chars)."
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("code")
    @classmethod
    def validate_code_format(cls, value: str) -> str:
        clean = value.strip()
        if not clean.isdigit() or len(clean) != 6:
            raise ValueError("Reset code must be 6 digits.")
        return clean


class ProfileUpdateRequest(BaseModel):
    """Payload for PATCH /v1/auth/profile."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str | None = Field(default=None, max_length=255, description="Updated display name.")
    avatar_url: str | None = Field(
        default=None, max_length=2_000_000, description="Updated avatar URL or base64 data URI."
    )
    location: str | None = Field(
        default=None,
        max_length=255,
        description="Updated address or location.",
        validation_alias=AliasChoices("location", "address"),
    )
    organization: str | None = Field(
        default=None,
        max_length=255,
        description="Updated company or organization.",
        validation_alias=AliasChoices("organization", "company"),
    )


class ReferralStatsResponse(BaseModel):
    """Referral dashboard metrics for current user."""

    referral_code: str = Field(description="User's unique referral code.")
    referral_link: str = Field(description="One-click sharable referral URL.")
    total_referrals: int = Field(description="Total registered users using this code.")
    pending_referrals: int = Field(description="Referrals awaiting qualification.")
    converted_referrals: int = Field(description="Converted referrals with bonus awarded.")
    total_earnings_usd: float = Field(description="Cumulative USD bonus credited to user.")
    reward_per_referral_usd: float = Field(
        default=5.0, description="Bonus paid per converted referral."
    )
    referee_bonus_usd: float = Field(default=5.0, description="Bonus awarded to invited referee.")


class ApiKeyCreatedResponse(BaseModel):
    """Response returned upon creation of a new API key containing plaintext secret."""

    id: int = Field(description="API key internal identifier.")
    name: str = Field(description="Label for this API key.")
    key: str = Field(description="Plaintext secret API key (displayed only once).")
    prefix: str = Field(description="Public key prefix.")
    status: str = Field(
        default="active", description="Key status ('active', 'expired', 'revoked')."
    )
    is_active: bool = Field(default=True, description="Whether key is active.")
    credit_balance: float = Field(description="Available balance allocated to this key in USD.")
    paid_balance: float = Field(default=0.0, description="Real paid balance in USD.")
    trial_balance: float = Field(default=0.0, description="Promotional/trial balance in USD.")
    rpm_limit: int = Field(description="Requests per minute rate limit.")
    tpm_limit: int = Field(description="Tokens per minute rate limit.")
    permissions: str = Field(description="Comma-separated permission scopes.")
    created_at: datetime = Field(description="Creation timestamp in UTC.")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp.")


class AuthResponse(BaseModel):
    """Authentication response payload containing user profile and JWT access token."""

    access_token: str = Field(description="Cryptographically signed JWT bearer token.")
    token_type: str = Field(default="bearer", description="Token type convention.")
    user: UserResponse = Field(description="Authenticated user details.")
    api_key: ApiKeyCreatedResponse | None = Field(
        default=None, description="Default API key generated upon registration if requested."
    )


class ApiKeyCreateRequest(BaseModel):
    """Payload for POST /v1/keys."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(default="default", max_length=100, description="Label for the API key.")
    credit_balance: float | None = Field(
        default=None, ge=0.0, description="Deprecated. New keys begin with no balance."
    )
    rpm_limit: int | None = Field(default=60, gt=0, description="Requests per minute rate limit.")
    tpm_limit: int | None = Field(default=60_000, gt=0, description="Tokens per minute rate limit.")
    permissions: str = Field(
        default="chat:completions,completions,models:read,usage:read",
        description="Comma-separated permission scopes.",
    )
    expires_in_days: int | None = Field(
        default=None, gt=0, description="Optional lifetime of key in days."
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Strip name and default to 'default' if empty or whitespace only."""
        stripped = value.strip()
        return stripped if stripped else "default"

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, value: str) -> str:
        """Ensure permissions string is non-empty and stripped."""
        stripped = value.strip()
        return stripped if stripped else "chat:completions,completions,models:read,usage:read"


class ApiKeyItemResponse(BaseModel):
    """API key summary item for listings."""

    id: int = Field(description="API key internal identifier.")
    name: str = Field(description="Human-readable label.")
    prefix: str = Field(description="Public key prefix.")
    masked_key: str = Field(description="Redacted key preview safe for display.")
    status: str = Field(description="Key status: 'active', 'expired', or 'revoked'.")
    is_active: bool = Field(description="Whether key is active.")
    credit_balance: float = Field(description="Current available balance in USD.")
    paid_balance: float = Field(default=0.0, description="Real paid balance in USD.")
    trial_balance: float = Field(default=0.0, description="Promotional/trial balance in USD.")
    rpm_limit: int = Field(description="Requests per minute rate limit.")
    tpm_limit: int = Field(description="Tokens per minute rate limit.")
    permissions: str = Field(description="Comma-separated permission scopes.")
    created_at: datetime = Field(description="Timestamp of creation in UTC.")
    last_used_at: datetime | None = Field(
        default=None, description="Timestamp of last inference call in UTC."
    )
    expires_at: datetime | None = Field(default=None, description="Expiration timestamp in UTC.")


class ApiKeyListResponse(BaseModel):
    """Response payload for GET /v1/keys."""

    object: Literal["list"] = "list"
    data: list[ApiKeyItemResponse] = Field(description="List of user's API keys.")
    total: int = Field(description="Total count of API keys belonging to user.")


class ApiKeyDeleteResponse(BaseModel):
    """Response payload for DELETE /v1/keys/{key_id}."""

    id: int = Field(description="ID of the revoked API key.")
    deleted: bool = Field(default=True, description="Deletion/revocation status flag.")
    status: str = Field(default="revoked", description="Updated status of the key.")
    message: str = Field(
        default="API key revoked successfully.", description="Status confirmation message."
    )
