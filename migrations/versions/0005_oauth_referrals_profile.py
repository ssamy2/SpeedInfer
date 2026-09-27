"""OAuth accounts, email verification codes, referrals, and user profile fields.

Revision ID: 0005_oauth_referrals_profile
Revises: 0004_workspace
Create Date: 2026-09-27 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_oauth_referrals_profile"
down_revision: str | None = "0004_workspace"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create OAuth, email verification, referral tables, and add profile columns to User."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # 1. Create oauth_account table
    if "oauth_account" not in existing_tables:
        op.create_table(
            "oauth_account",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey("user.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.Column("provider_user_id", sa.String(length=128), nullable=False),
            sa.Column("provider_email", sa.String(length=255), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_oauth_account_user_id", "oauth_account", ["user_id"])
        op.create_index("ix_oauth_account_provider", "oauth_account", ["provider"])
        op.create_index("ix_oauth_account_provider_user_id", "oauth_account", ["provider_user_id"])
        op.create_index(
            "ix_oauth_provider_uid",
            "oauth_account",
            ["provider", "provider_user_id"],
            unique=True,
        )

    # 2. Create email_verification_code table
    if "email_verification_code" not in existing_tables:
        op.create_table(
            "email_verification_code",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("email", sa.String(length=255), nullable=False),
            sa.Column("code", sa.String(length=8), nullable=False),
            sa.Column(
                "purpose", sa.String(length=32), nullable=False, server_default="registration"
            ),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("is_used", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_email_verification_code_email", "email_verification_code", ["email"])
        op.create_index("ix_email_verification_code_code", "email_verification_code", ["code"])
        op.create_index(
            "ix_email_code_purpose",
            "email_verification_code",
            ["email", "code", "purpose"],
        )

    # 3. Create referral table
    if "referral" not in existing_tables:
        op.create_table(
            "referral",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column(
                "referrer_id",
                sa.Integer(),
                sa.ForeignKey("user.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "referred_id",
                sa.Integer(),
                sa.ForeignKey("user.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
            ),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
            sa.Column("referee_bonus_awarded", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("referrer_reward_awarded", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("reward_amount", sa.Float(), nullable=False, server_default="5.0"),
            sa.Column("device_fingerprint", sa.String(length=128), nullable=True),
            sa.Column("signup_ip_hash", sa.String(length=64), nullable=True),
            sa.Column("fraud_flag", sa.Boolean(), nullable=False, server_default="0"),
            sa.Column("fraud_reason", sa.String(length=255), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("rewarded_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_referral_referrer_id", "referral", ["referrer_id"])
        op.create_index("ix_referral_referred_id", "referral", ["referred_id"])
        op.create_index("ix_referral_created_at", "referral", ["created_at"])
        op.create_index("ix_referral_device_fingerprint", "referral", ["device_fingerprint"])
        op.create_index("ix_referral_signup_ip_hash", "referral", ["signup_ip_hash"])

    # 4. Add new columns to user table if not present
    existing_user_cols = {col["name"] for col in inspector.get_columns("user")}
    existing_user_indexes = {idx["name"] for idx in inspector.get_indexes("user")}

    with op.batch_alter_table("user") as batch_op:
        if "referral_code" not in existing_user_cols:
            batch_op.add_column(sa.Column("referral_code", sa.String(length=32), nullable=True))
        if "referred_by_id" not in existing_user_cols:
            batch_op.add_column(
                sa.Column(
                    "referred_by_id",
                    sa.Integer(),
                    sa.ForeignKey("user.id", name="fk_user_referred_by_id"),
                    nullable=True,
                )
            )
        if "signup_ip_hash" not in existing_user_cols:
            batch_op.add_column(sa.Column("signup_ip_hash", sa.String(length=64), nullable=True))
        if "device_fingerprint" not in existing_user_cols:
            batch_op.add_column(
                sa.Column("device_fingerprint", sa.String(length=128), nullable=True)
            )
        if "referral_reward_claimed" not in existing_user_cols:
            batch_op.add_column(
                sa.Column(
                    "referral_reward_claimed",
                    sa.Boolean(),
                    nullable=False,
                    server_default="0",
                )
            )
        if "is_verified" not in existing_user_cols:
            batch_op.add_column(
                sa.Column(
                    "is_verified",
                    sa.Boolean(),
                    nullable=False,
                    server_default="0",
                )
            )
        if "email_verified_at" not in existing_user_cols:
            batch_op.add_column(sa.Column("email_verified_at", sa.DateTime(), nullable=True))
        if "avatar_url" not in existing_user_cols:
            batch_op.add_column(sa.Column("avatar_url", sa.Text(), nullable=True))
        if "location" not in existing_user_cols:
            batch_op.add_column(sa.Column("location", sa.String(length=255), nullable=True))
        if "organization" not in existing_user_cols:
            batch_op.add_column(sa.Column("organization", sa.String(length=255), nullable=True))

        if "ix_user_referral_code" not in existing_user_indexes:
            batch_op.create_index("ix_user_referral_code", ["referral_code"], unique=True)
        if "ix_user_referred_by_id" not in existing_user_indexes:
            batch_op.create_index("ix_user_referred_by_id", ["referred_by_id"])
        if "ix_user_signup_ip_hash" not in existing_user_indexes:
            batch_op.create_index("ix_user_signup_ip_hash", ["signup_ip_hash"])
        if "ix_user_device_fingerprint" not in existing_user_indexes:
            batch_op.create_index("ix_user_device_fingerprint", ["device_fingerprint"])

    # Populate any existing user records with a unique referral code
    if bind.dialect.name == "sqlite":
        op.execute(
            sa.text(
                "UPDATE \"user\" SET referral_code = 'REF' || hex(randomblob(4)) "
                "WHERE referral_code IS NULL"
            )
        )
    else:
        op.execute(
            sa.text(
                "UPDATE \"user\" SET referral_code = "
                "'REF' || upper(substring(md5(random()::text), 1, 8)) "
                "WHERE referral_code IS NULL"
            )
        )


def downgrade() -> None:
    """Drop referral, email verification, oauth tables and user columns."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "user" in existing_tables:
        existing_user_cols = {col["name"] for col in inspector.get_columns("user")}
        existing_user_indexes = {idx["name"] for idx in inspector.get_indexes("user")}

        with op.batch_alter_table("user") as batch_op:
            if "ix_user_device_fingerprint" in existing_user_indexes:
                batch_op.drop_index("ix_user_device_fingerprint")
            if "ix_user_signup_ip_hash" in existing_user_indexes:
                batch_op.drop_index("ix_user_signup_ip_hash")
            if "ix_user_referred_by_id" in existing_user_indexes:
                batch_op.drop_index("ix_user_referred_by_id")
            if "ix_user_referral_code" in existing_user_indexes:
                batch_op.drop_index("ix_user_referral_code")

            for col in (
                "organization",
                "location",
                "avatar_url",
                "email_verified_at",
                "is_verified",
                "referral_reward_claimed",
                "device_fingerprint",
                "signup_ip_hash",
                "referred_by_id",
                "referral_code",
            ):
                if col in existing_user_cols:
                    batch_op.drop_column(col)

    if "referral" in existing_tables:
        op.drop_table("referral")
    if "email_verification_code" in existing_tables:
        op.drop_table("email_verification_code")
    if "oauth_account" in existing_tables:
        op.drop_table("oauth_account")
