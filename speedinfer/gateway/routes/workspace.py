"""Private workspace resources and explicitly simulated model workflows.

Objects are bounded database blobs for the local preview, not an S3 service.
No simulation operation calls a GPU provider or changes an API credit balance.
"""

import json
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from pydantic import Field as PydanticField
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from speedinfer.core.security import get_current_user
from speedinfer.database.models import User
from speedinfer.database.session import get_session
from speedinfer.database.workspace import WorkspaceObject, WorkspaceResource

router = APIRouter(prefix="/v1/workspace", tags=["Workspace"])
MAX_OBJECT_BYTES = 20 * 1024 * 1024
MAX_USER_BYTES = 100 * 1024 * 1024
PRICES = [
    {
        "sku": "demo-small",
        "name": "Dedicated Endpoint — Standard (24GB)",
        "tier": "Dedicated Inference Capacity",
        "memory_gb": 24,
        "hourly_usd": 0.50,
        "pricing_model": "dedicated-capacity",
    },
    {
        "sku": "demo-medium",
        "name": "Dedicated Endpoint — High-Throughput (48GB)",
        "tier": "Dedicated Inference Capacity",
        "memory_gb": 48,
        "hourly_usd": 1.00,
        "pricing_model": "dedicated-capacity",
    },
    {
        "sku": "demo-large",
        "name": "Dedicated Endpoint — Enterprise Ultra (80GB)",
        "tier": "Dedicated Inference Capacity",
        "memory_gb": 80,
        "hourly_usd": 2.00,
        "pricing_model": "dedicated-capacity",
    },
]


UserDep = Annotated[User, Depends(get_current_user)]
SessionDep = Annotated[Session, Depends(get_session)]


class ResourceRequest(BaseModel):
    kind: Literal["project", "bucket", "training", "deployment", "evaluation"]
    name: str = PydanticField(min_length=1, max_length=100, pattern=r".*\S.*")
    request_id: str = PydanticField(min_length=8, max_length=100)
    project_id: str | None = None
    dataset_id: str | None = None
    weights_id: str | None = None
    artifact_id: str | None = None
    model: str = PydanticField(default="Qwen/Qwen2.5-7B-Instruct", max_length=200)
    method: Literal["lora", "qlora"] = "qlora"
    sku: Literal["demo-small", "demo-medium", "demo-large"] = "demo-small"
    hours: float = PydanticField(default=1, gt=0, le=168)
    replicas: int = PydanticField(default=1, ge=1, le=4)


class ActionRequest(BaseModel):
    action: Literal["start", "stop", "complete", "fail", "cancel"]


class EstimateRequest(BaseModel):
    sku: Literal["demo-small", "demo-medium", "demo-large"]
    hours: float = PydanticField(gt=0, le=168)
    replicas: int = PydanticField(default=1, ge=1, le=4)


def owned(session: Session, user: User, resource_id: str, kind: str | None = None):
    item = session.get(WorkspaceResource, resource_id)
    if not item or item.user_id != user.id or (kind and item.kind != kind):
        raise HTTPException(404, "Resource not found")
    return item


def object_owned(session: Session, user: User, object_id: str):
    item = session.get(WorkspaceObject, object_id)
    if not item or item.user_id != user.id:
        raise HTTPException(404, "Object not found")
    return item


def public_resource(item):
    return {
        "id": item.id,
        "kind": item.kind,
        "name": item.name,
        "status": item.status,
        "created_at": item.created_at,
        "data": json.loads(item.data_json),
    }


def public_object(item):
    return {
        "id": item.id,
        "bucket_id": item.bucket_id,
        "name": item.name,
        "purpose": item.purpose,
        "size": item.size,
        "created_at": item.created_at,
    }


def estimate(sku: str, hours: float, replicas: int):
    price = next(p for p in PRICES if p["sku"] == sku)
    return {
        "currency": "USD",
        "mode": "simulation",
        "tier": "Dedicated Inference Capacity",
        "version": "demo-2026-09-27",
        "hourly_usd": price["hourly_usd"],
        "hours": hours,
        "replicas": replicas,
        "total_usd": round(price["hourly_usd"] * hours * replicas, 4),
        "charge_usd": 0,
        "excludes": ["storage", "network"],
    }


@router.get("/pricing")
def pricing():
    return {
        "items": PRICES,
        "currency": "USD",
        "mode": "simulation",
        "version": "demo-2026-09-27",
        "source": "Dedicated endpoint capacity estimates",
        "max_object_bytes": MAX_OBJECT_BYTES,
        "max_user_bytes": MAX_USER_BYTES,
    }


@router.post("/estimate")
def estimate_resource(payload: EstimateRequest, user: UserDep):
    return estimate(payload.sku, payload.hours, payload.replicas)


@router.get("/resources")
def resources(user: UserDep, session: SessionDep):
    items = session.exec(
        select(WorkspaceResource)
        .where(WorkspaceResource.user_id == user.id)
        .order_by(WorkspaceResource.created_at.desc())
    ).all()
    # Do not load object contents into the workspace listing.
    objects = session.exec(
        select(
            WorkspaceObject.id,
            WorkspaceObject.bucket_id,
            WorkspaceObject.name,
            WorkspaceObject.purpose,
            WorkspaceObject.size,
            WorkspaceObject.created_at,
        ).where(WorkspaceObject.user_id == user.id)
    ).all()
    return {
        "resources": [public_resource(item) for item in items],
        "objects": [dict(row._mapping) for row in objects],
    }


@router.post("/resources", status_code=201)
def create_resource(payload: ResourceRequest, user: UserDep, session: SessionDep):
    # Serialize account mutations before checking quotas and references.
    session.execute(update(User).where(User.id == user.id).values(updated_at=User.updated_at))
    existing = session.exec(
        select(WorkspaceResource).where(
            WorkspaceResource.user_id == user.id, WorkspaceResource.request_id == payload.request_id
        )
    ).first()
    if existing:
        return public_resource(existing)
    if (
        len(
            session.exec(
                select(WorkspaceResource.id).where(WorkspaceResource.user_id == user.id)
            ).all()
        )
        >= 200
    ):
        raise HTTPException(409, "Preview limit: 200 resources per account")
    data = payload.model_dump(exclude={"kind", "name", "request_id"})
    if payload.project_id:
        owned(session, user, payload.project_id, "project")
    if payload.kind in {"training", "deployment", "evaluation"} and not payload.project_id:
        raise HTTPException(422, "Select a project first")
    if payload.dataset_id:
        dataset = object_owned(session, user, payload.dataset_id)
        if dataset.purpose != "dataset":
            raise HTTPException(422, "Select a dataset object")
    if payload.weights_id:
        weights = object_owned(session, user, payload.weights_id)
        if weights.purpose != "weights":
            raise HTTPException(422, "Select a model weights object")
    if payload.kind == "training" and not payload.dataset_id:
        raise HTTPException(422, "A training dataset is required")
    if payload.artifact_id:
        artifact = owned(session, user, payload.artifact_id, "training")
        if artifact.status != "succeeded":
            raise HTTPException(422, "Select a completed training simulation")
        if json.loads(artifact.data_json).get("project_id") != payload.project_id:
            raise HTTPException(422, "Artifact belongs to a different project")
    if payload.kind == "evaluation" and not payload.artifact_id:
        raise HTTPException(422, "Select a completed training simulation")
    simulated = payload.kind in {"training", "deployment", "evaluation"}
    data.update(
        {
            "is_simulated": simulated,
            "estimate": estimate(payload.sku, payload.hours, payload.replicas)
            if simulated
            else None,
            "events": [
                {
                    "at": datetime.now(UTC).isoformat(),
                    "message": "Simulation created. No GPU allocated."
                    if simulated
                    else "Resource created.",
                }
            ],
        }
    )
    item = WorkspaceResource(
        user_id=user.id,
        kind=payload.kind,
        name=payload.name.strip(),
        request_id=payload.request_id,
        data_json=json.dumps(data),
        status="queued" if simulated else "ready",
    )
    session.add(item)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        existing = session.exec(
            select(WorkspaceResource).where(
                WorkspaceResource.user_id == user.id,
                WorkspaceResource.request_id == payload.request_id,
            )
        ).first()
        if existing:
            return public_resource(existing)
        raise
    session.refresh(item)
    return public_resource(item)


@router.post("/resources/{resource_id}/actions")
def resource_action(resource_id: str, payload: ActionRequest, user: UserDep, session: SessionDep):
    session.execute(update(User).where(User.id == user.id).values(updated_at=User.updated_at))
    item = owned(session, user, resource_id)
    if item.kind not in {"training", "deployment", "evaluation"}:
        raise HTTPException(409, "This resource has no execution lifecycle")
    transitions = {
        "start": ({"queued", "stopped"}, "running"),
        "stop": ({"running"}, "stopped"),
        "cancel": ({"queued", "running", "stopped"}, "cancelled"),
        "fail": ({"queued", "running"}, "failed"),
        "complete": ({"running"}, "succeeded"),
    }
    allowed, target = transitions[payload.action]
    if item.status not in allowed or (payload.action == "stop" and item.kind != "deployment"):
        raise HTTPException(409, "Action is not available in this state")
    if payload.action == "complete" and item.kind == "deployment":
        raise HTTPException(409, "Stop a deployment instead of completing it")
    data = json.loads(item.data_json)
    data["events"].append(
        {
            "at": datetime.now(UTC).isoformat(),
            "message": f"Simulation: {item.status} → {target}. No charge.",
        }
    )
    data["events"] = data["events"][-200:]
    if target == "succeeded":
        data["result"] = "Simulated result only. No trained weights or measured metrics generated."
    item.status = target
    item.data_json = json.dumps(data)
    session.add(item)
    session.commit()
    session.refresh(item)
    return public_resource(item)


@router.post("/buckets/{bucket_id}/objects", status_code=201)
async def upload_object(
    bucket_id: str,
    request: Request,
    user: UserDep,
    session: SessionDep,
    name: str,
    purpose: Literal["dataset", "weights", "other"] = "other",
):
    owned(session, user, bucket_id, "bucket")
    if not name.strip() or len(name) > 200 or any(c in name for c in "\\/\r\n\x00"):
        raise HTTPException(422, "Use a file name without folders or control characters")
    sizes = session.exec(
        select(WorkspaceObject.size).where(WorkspaceObject.user_id == user.id)
    ).all()
    if len(sizes) >= 100:
        raise HTTPException(413, "Preview limit: 100 files per account")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_OBJECT_BYTES or sum(sizes) + len(body) > MAX_USER_BYTES:
            raise HTTPException(413, "Preview storage limit: 20 MiB per file, 100 MiB per account")
    if not body:
        raise HTTPException(422, "Empty files are not supported")
    if purpose == "dataset":
        if not name.lower().endswith(".jsonl"):
            raise HTTPException(422, "Training preview accepts UTF-8 JSONL datasets")
        try:
            rows = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
            if not rows or any(
                not isinstance(row, dict)
                or not (isinstance(row.get("text"), str) or isinstance(row.get("messages"), list))
                for row in rows
            ):
                raise ValueError
        except (ValueError, UnicodeError):
            raise HTTPException(
                422, "Each JSONL row needs a text string or messages array"
            ) from None
    if purpose == "weights" and not name.lower().endswith((".safetensors", ".gguf", ".bin")):
        raise HTTPException(422, "Use a .safetensors, .gguf or .bin file; files are never executed")
    # Recheck quota under an account write lock after reading the bounded body.
    session.execute(update(User).where(User.id == user.id).values(updated_at=User.updated_at))
    current_sizes = session.exec(
        select(WorkspaceObject.size).where(WorkspaceObject.user_id == user.id)
    ).all()
    if len(current_sizes) >= 100 or sum(current_sizes) + len(body) > MAX_USER_BYTES:
        raise HTTPException(413, "Account storage limit reached")
    item = WorkspaceObject(
        user_id=user.id,
        bucket_id=bucket_id,
        name=name.strip(),
        purpose=purpose,
        size=len(body),
        content=bytes(body),
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return public_object(item)


@router.get("/objects/{object_id}")
def download_object(object_id: str, user: UserDep, session: SessionDep):
    item = object_owned(session, user, object_id)
    return Response(
        item.content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "attachment",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


@router.delete("/objects/{object_id}", status_code=204)
def delete_object(object_id: str, user: UserDep, session: SessionDep):
    session.execute(update(User).where(User.id == user.id).values(updated_at=User.updated_at))
    item = object_owned(session, user, object_id)
    for resource in session.exec(
        select(WorkspaceResource).where(WorkspaceResource.user_id == user.id)
    ).all():
        data = json.loads(resource.data_json)
        if object_id in (data.get("dataset_id"), data.get("weights_id")):
            raise HTTPException(
                409, "File is referenced by a workflow; keep it for reproducibility"
            )
    session.delete(item)
    session.commit()
