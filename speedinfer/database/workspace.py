"""Private preview resources, bounded file objects and policy acknowledgements."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, LargeBinary, UniqueConstraint
from sqlmodel import Field, SQLModel


class WorkspaceResource(SQLModel, table=True):
    __tablename__ = "workspace_resource"
    __table_args__ = (UniqueConstraint("user_id", "request_id"),)
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    kind: str = Field(index=True)
    name: str
    status: str = "ready"
    data_json: str = "{}"
    request_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class WorkspaceObject(SQLModel, table=True):
    __tablename__ = "workspace_object"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    bucket_id: str = Field(foreign_key="workspace_resource.id", index=True)
    name: str
    purpose: str
    size: int
    content: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PolicyAcceptance(SQLModel, table=True):
    __tablename__ = "policy_acceptance"
    user_id: int = Field(foreign_key="user.id", primary_key=True)
    version: str = Field(primary_key=True)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
