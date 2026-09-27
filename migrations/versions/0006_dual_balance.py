"""Dual-balance support (trial_balance and paid_balance) for ApiKey.

Revision ID: 0006_dual_balance
Revises: 0005_oauth_referrals_profile
Create Date: 2026-09-27 13:15:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_dual_balance"
down_revision: str | None = "0005_oauth_referrals_profile"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add trial_balance and paid_balance columns to apikey table and backfill."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "apikey" in existing_tables:
        existing_cols = {col["name"] for col in inspector.get_columns("apikey")}

        with op.batch_alter_table("apikey") as batch_op:
            if "trial_balance" not in existing_cols:
                batch_op.add_column(
                    sa.Column("trial_balance", sa.Float(), nullable=False, server_default="0.0")
                )
            if "paid_balance" not in existing_cols:
                batch_op.add_column(
                    sa.Column("paid_balance", sa.Float(), nullable=False, server_default="0.0")
                )

        # Backfill existing keys where paid_balance is 0 and credit_balance > 0
        op.execute(
            sa.text(
                "UPDATE apikey SET paid_balance = credit_balance "
                "WHERE (trial_balance + paid_balance) = 0.0 AND credit_balance > 0.0"
            )
        )


def downgrade() -> None:
    """Drop trial_balance and paid_balance columns from apikey table."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "apikey" in existing_tables:
        existing_cols = {col["name"] for col in inspector.get_columns("apikey")}

        with op.batch_alter_table("apikey") as batch_op:
            if "paid_balance" in existing_cols:
                batch_op.drop_column("paid_balance")
            if "trial_balance" in existing_cols:
                batch_op.drop_column("trial_balance")
