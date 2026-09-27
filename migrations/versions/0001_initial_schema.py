"""Initial database schema creation.

Revision ID: 0001
Revises:
Create Date: 2026-09-26 00:00:00.000000

Creates initial tables:
- user: account holder with email uniqueness and active/admin flags
- apikey: hashed API keys with prefix, balance, limits, and permissions
- usageledger: immutable audit log of inference requests and token costs
- modelversion: model registry entries and pricing definitions
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all initial tables, indexes, and constraints."""
    # 1. Create table 'user'
    op.create_table(
        "user",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user")),
    )
    op.create_index(op.f("ix_user_email"), "user", ["email"], unique=True)

    # 2. Create table 'apikey'
    op.create_table(
        "apikey",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False, server_default="default"),
        sa.Column("key_hash", sa.String(length=64), nullable=False),
        sa.Column("prefix", sa.String(length=32), nullable=False),
        sa.Column(
            "permissions",
            sa.String(length=512),
            nullable=False,
            server_default="chat:completions,completions,models:read,usage:read",
        ),
        sa.Column("credit_balance", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("rpm_limit", sa.Integer(), nullable=False, server_default=sa.text("60")),
        sa.Column("tpm_limit", sa.Integer(), nullable=False, server_default=sa.text("60000")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("credit_balance >= 0.0", name=op.f("ck_apikey_credit_balance_positive")),
        sa.CheckConstraint("rpm_limit > 0", name=op.f("ck_apikey_rpm_limit_positive")),
        sa.CheckConstraint("tpm_limit > 0", name=op.f("ck_apikey_tpm_limit_positive")),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["user.id"],
            name=op.f("fk_apikey_user_id_user"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_apikey")),
    )
    op.create_index(op.f("ix_apikey_user_id"), "apikey", ["user_id"], unique=False)
    op.create_index(op.f("ix_apikey_key_hash"), "apikey", ["key_hash"], unique=True)
    op.create_index(op.f("ix_apikey_prefix"), "apikey", ["prefix"], unique=False)

    # 3. Create table 'usageledger'
    op.create_table(
        "usageledger",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("api_key_id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("total_cost", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("ttft_ms", sa.Float(), nullable=True),
        sa.Column("status_code", sa.Integer(), nullable=False, server_default=sa.text("200")),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "prompt_tokens >= 0", name=op.f("ck_usageledger_prompt_tokens_non_negative")
        ),
        sa.CheckConstraint(
            "completion_tokens >= 0",
            name=op.f("ck_usageledger_completion_tokens_non_negative"),
        ),
        sa.CheckConstraint(
            "total_tokens >= 0", name=op.f("ck_usageledger_total_tokens_non_negative")
        ),
        sa.CheckConstraint(
            "total_cost >= 0.0", name=op.f("ck_usageledger_total_cost_non_negative")
        ),
        sa.CheckConstraint(
            "latency_ms >= 0.0", name=op.f("ck_usageledger_latency_ms_non_negative")
        ),
        sa.CheckConstraint(
            "status_code >= 100 AND status_code <= 599",
            name=op.f("ck_usageledger_status_code_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["api_key_id"],
            ["apikey.id"],
            name=op.f("fk_usageledger_api_key_id_apikey"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_usageledger")),
    )
    op.create_index(op.f("ix_usageledger_api_key_id"), "usageledger", ["api_key_id"], unique=False)
    op.create_index(op.f("ix_usageledger_request_id"), "usageledger", ["request_id"], unique=True)
    op.create_index(op.f("ix_usageledger_model"), "usageledger", ["model"], unique=False)
    op.create_index(op.f("ix_usageledger_created_at"), "usageledger", ["created_at"], unique=False)
    op.create_index(
        op.f("ix_usageledger_api_key_created_at"),
        "usageledger",
        ["api_key_id", "created_at"],
        unique=False,
    )

    # 4. Create table 'modelversion'
    op.create_table(
        "modelversion",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("base_model_path", sa.String(length=255), nullable=False),
        sa.Column("adapter_path", sa.String(length=512), nullable=True),
        sa.Column(
            "lifecycle_status",
            sa.String(length=32),
            nullable=False,
            server_default="active",
        ),
        sa.Column("mlflow_run_id", sa.String(length=64), nullable=True),
        sa.Column("mlflow_model_version", sa.String(length=32), nullable=True),
        sa.Column("eval_metrics", sa.Text(), nullable=True),
        sa.Column("context_length", sa.Integer(), nullable=False, server_default=sa.text("32768")),
        sa.Column(
            "prompt_price_per_million",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.20"),
        ),
        sa.Column(
            "completion_price_per_million",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0.60"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "context_length > 0", name=op.f("ck_modelversion_context_length_positive")
        ),
        sa.CheckConstraint(
            "prompt_price_per_million >= 0.0",
            name=op.f("ck_modelversion_prompt_price_non_negative"),
        ),
        sa.CheckConstraint(
            "completion_price_per_million >= 0.0",
            name=op.f("ck_modelversion_completion_price_non_negative"),
        ),
        sa.CheckConstraint(
            "lifecycle_status IN ('active', 'evaluating', 'deprecated', "
            "'training', 'staging', 'failed_evaluation')",
            name=op.f("ck_modelversion_lifecycle_status_valid"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_modelversion")),
    )
    op.create_index(op.f("ix_modelversion_name"), "modelversion", ["name"], unique=True)
    op.create_index(
        op.f("ix_modelversion_lifecycle_status"),
        "modelversion",
        ["lifecycle_status"],
        unique=False,
    )


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_table("modelversion")
    op.drop_table("usageledger")
    op.drop_table("apikey")
    op.drop_table("user")
