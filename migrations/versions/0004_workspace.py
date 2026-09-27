"""Private preview storage and simulated managed model workflows."""

import sqlalchemy as sa
from alembic import op

revision = "0004_workspace"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "policy_acceptance",
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), primary_key=True),
        sa.Column("version", sa.String(), primary_key=True),
        sa.Column("created_at", sa.String(), nullable=False),
    )
    op.create_table(
        "trial_credit_grant",
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), primary_key=True),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    # Existing accounts cannot claim a fresh trial by deleting all their old keys.
    op.execute(
        sa.text(
            "INSERT INTO trial_credit_grant (user_id, amount, created_at) "
            'SELECT id, 0, CURRENT_TIMESTAMP FROM "user"'
        )
    )
    op.create_table(
        "workspace_resource",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("data_json", sa.String(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.UniqueConstraint("user_id", "request_id"),
    )
    op.create_index("ix_workspace_resource_user_id", "workspace_resource", ["user_id"])
    op.create_index("ix_workspace_resource_kind", "workspace_resource", ["kind"])
    op.create_table(
        "workspace_object",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("bucket_id", sa.String(), sa.ForeignKey("workspace_resource.id"), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("purpose", sa.String(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
    )
    op.create_index("ix_workspace_object_user_id", "workspace_object", ["user_id"])
    op.create_index("ix_workspace_object_bucket_id", "workspace_object", ["bucket_id"])


def downgrade():
    op.drop_table("workspace_object")
    op.drop_table("workspace_resource")
    op.drop_table("trial_credit_grant")
    op.drop_table("policy_acceptance")
