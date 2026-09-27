"""Add password_hash to user model.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-26 01:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# Revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add password_hash column to user table."""
    with op.batch_alter_table("user") as batch_op:
        batch_op.add_column(sa.Column("password_hash", sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Remove password_hash column from user table."""
    with op.batch_alter_table("user") as batch_op:
        batch_op.drop_column("password_hash")
