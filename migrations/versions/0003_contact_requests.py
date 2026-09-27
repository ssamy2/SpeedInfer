"""Store sales and support requests.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "contactrequest",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("email", sa.String(254), nullable=False),
        sa.Column("company", sa.String(120), nullable=False),
        sa.Column("subject", sa.String(160), nullable=False),
        sa.Column("message", sa.String(3000), nullable=False),
        sa.Column("client_fingerprint", sa.String(64), nullable=False),
        sa.Column("delivery_status", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    for column in ("email", "client_fingerprint", "created_at"):
        op.create_index(f"ix_contactrequest_{column}", "contactrequest", [column])


def downgrade():
    op.drop_table("contactrequest")
