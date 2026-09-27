"""Durable inference reservations, independent of Redis cache lifetime."""

import sqlalchemy as sa
from alembic import op

revision = "0007_inference_reservations"
down_revision = "0006_dual_balance"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "inference_reservation",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("apikey.id"), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("trial", sa.Float(), nullable=False),
        sa.Column("paid", sa.Float(), nullable=False),
        sa.Column("state", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
    )
    op.create_index("ix_inference_reservation_api_key_id", "inference_reservation", ["api_key_id"])
    op.create_index("ix_inference_reservation_state", "inference_reservation", ["state"])


def downgrade():
    op.drop_table("inference_reservation")
