"""Developer API for Buckets and Object Storage.

Provides:
- POST /v1/buckets: Create a storage bucket.
- GET /v1/buckets: List all buckets owned by the authenticated caller.
- GET /v1/buckets/{bucket_id}: Retrieve bucket details.
- DELETE /v1/buckets/{bucket_id}: Delete a bucket and its objects.
- POST /v1/buckets/{bucket_id}/objects: Upload an object to a bucket.
- GET /v1/buckets/{bucket_id}/objects: List objects in a bucket.
- GET /v1/buckets/{bucket_id}/objects/{object_name:path}: Download object content.
- DELETE /v1/buckets/{bucket_id}/objects/{object_name:path}: Delete an object.
"""

from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from speedinfer.core.auth import check_scope_permission
from speedinfer.database.models import ApiKey, BucketObjectRecord, BucketRecord, User
from speedinfer.database.session import get_session
from speedinfer.gateway.routes.models import get_caller_auth

router = APIRouter(prefix="/v1/buckets", tags=["Storage & Buckets"])

MAX_OBJECT_BYTES = 500 * 1024 * 1024  # 500 MiB per object


class BucketCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[a-zA-Z0-9_\-\.]+$")
    description: str | None = Field(default=None, max_length=512)


class BucketObjectResponse(BaseModel):
    id: str
    bucket_id: str
    name: str
    size: int
    content_type: str
    created_at: str


class BucketResponse(BaseModel):
    id: str
    name: str
    description: str | None
    created_at: str
    object_count: int = 0
    total_bytes: int = 0


class BucketListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[BucketResponse]


class ObjectListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[BucketObjectResponse]


def _resolve_user_id(caller: ApiKey | User) -> int:
    return caller.user_id if isinstance(caller, ApiKey) else caller.id


@router.post(
    "",
    response_model=BucketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a storage bucket",
)
@router.post(
    "/",
    response_model=BucketResponse,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_bucket(
    payload: BucketCreateRequest,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> BucketResponse:
    """Create a new object storage bucket."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:write"):
        raise HTTPException(status_code=403, detail="Missing storage:write permission.")

    user_id = _resolve_user_id(caller)
    existing = session.exec(
        select(BucketRecord).where(
            BucketRecord.user_id == user_id,
            BucketRecord.name == payload.name.strip(),
        )
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Bucket with name '{payload.name}' already exists in your account.",
        )

    bucket = BucketRecord(
        user_id=user_id,
        name=payload.name.strip(),
        description=payload.description,
    )
    session.add(bucket)
    session.commit()
    session.refresh(bucket)

    return BucketResponse(
        id=bucket.id,
        name=bucket.name,
        description=bucket.description,
        created_at=bucket.created_at.isoformat(),
        object_count=0,
        total_bytes=0,
    )


@router.get(
    "",
    response_model=BucketListResponse,
    summary="List buckets",
)
@router.get(
    "/",
    response_model=BucketListResponse,
    include_in_schema=False,
)
async def list_buckets(
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> BucketListResponse:
    """List all buckets belonging to the caller."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:read"):
        raise HTTPException(status_code=403, detail="Missing storage:read permission.")

    user_id = _resolve_user_id(caller)
    buckets = session.exec(
        select(BucketRecord)
        .where(BucketRecord.user_id == user_id)
        .order_by(BucketRecord.created_at.desc())
    ).all()

    items = []
    for b in buckets:
        objs = session.exec(
            select(BucketObjectRecord.size).where(BucketObjectRecord.bucket_id == b.id)
        ).all()
        items.append(
            BucketResponse(
                id=b.id,
                name=b.name,
                description=b.description,
                created_at=b.created_at.isoformat(),
                object_count=len(objs),
                total_bytes=sum(objs),
            )
        )
    return BucketListResponse(data=items)


@router.get(
    "/{bucket_id}",
    response_model=BucketResponse,
    summary="Retrieve bucket",
)
async def get_bucket(
    bucket_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> BucketResponse:
    """Retrieve details for a specific bucket."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:read"):
        raise HTTPException(status_code=403, detail="Missing storage:read permission.")

    user_id = _resolve_user_id(caller)
    bucket = session.get(BucketRecord, bucket_id)
    if not bucket or bucket.user_id != user_id:
        # Check by name as fallback
        bucket = session.exec(
            select(BucketRecord).where(
                BucketRecord.user_id == user_id,
                BucketRecord.name == bucket_id,
            )
        ).first()

    if not bucket or bucket.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_id}' not found.")

    objs = session.exec(
        select(BucketObjectRecord.size).where(BucketObjectRecord.bucket_id == bucket.id)
    ).all()

    return BucketResponse(
        id=bucket.id,
        name=bucket.name,
        description=bucket.description,
        created_at=bucket.created_at.isoformat(),
        object_count=len(objs),
        total_bytes=sum(objs),
    )


@router.delete(
    "/{bucket_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete bucket",
)
async def delete_bucket(
    bucket_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
):
    """Delete a bucket and all its contained objects."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:write"):
        raise HTTPException(status_code=403, detail="Missing storage:write permission.")

    user_id = _resolve_user_id(caller)
    bucket = session.get(BucketRecord, bucket_id)
    if not bucket or bucket.user_id != user_id:
        bucket = session.exec(
            select(BucketRecord).where(
                BucketRecord.user_id == user_id,
                BucketRecord.name == bucket_id,
            )
        ).first()

    if not bucket or bucket.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_id}' not found.")

    # Delete all objects in bucket
    objs = session.exec(
        select(BucketObjectRecord).where(BucketObjectRecord.bucket_id == bucket.id)
    ).all()
    for obj in objs:
        session.delete(obj)

    session.delete(bucket)
    session.commit()
    return None


@router.post(
    "/{bucket_id}/objects",
    response_model=BucketObjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload object to bucket",
)
async def upload_bucket_object(
    bucket_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
    request: Request,
    name: str | None = None,
    file: UploadFile | None = None,
) -> BucketObjectResponse:
    """Upload an object into a bucket."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:write"):
        raise HTTPException(status_code=403, detail="Missing storage:write permission.")

    user_id = _resolve_user_id(caller)
    bucket = session.get(BucketRecord, bucket_id)
    if not bucket or bucket.user_id != user_id:
        bucket = session.exec(
            select(BucketRecord).where(
                BucketRecord.user_id == user_id,
                BucketRecord.name == bucket_id,
            )
        ).first()

    if not bucket or bucket.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_id}' not found.")

    if file is not None:
        object_name = (name or file.filename or "unnamed_object").strip()
        content = await file.read()
        content_type = file.content_type or "application/octet-stream"
    else:
        # Read from raw stream
        if not name:
            raise HTTPException(
                status_code=400,
                detail="Query parameter 'name' is required when uploading raw data.",
            )
        object_name = name.strip()
        content = await request.body()
        content_type = request.headers.get("content-type", "application/octet-stream")

    if not content:
        raise HTTPException(status_code=400, detail="Object content cannot be empty.")
    if len(content) > MAX_OBJECT_BYTES:
        raise HTTPException(
            status_code=413, detail=f"Object size exceeds {MAX_OBJECT_BYTES // (1024 * 1024)} MiB."
        )

    # Check for existing object with same name in bucket
    existing = session.exec(
        select(BucketObjectRecord).where(
            BucketObjectRecord.bucket_id == bucket.id,
            BucketObjectRecord.name == object_name,
        )
    ).first()

    if existing:
        existing.content = content
        existing.size = len(content)
        existing.content_type = content_type
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return BucketObjectResponse(
            id=existing.id,
            bucket_id=bucket.id,
            name=existing.name,
            size=existing.size,
            content_type=existing.content_type,
            created_at=existing.created_at.isoformat(),
        )

    obj = BucketObjectRecord(
        user_id=user_id,
        bucket_id=bucket.id,
        name=object_name,
        size=len(content),
        content_type=content_type,
        content=content,
    )
    session.add(obj)
    session.commit()
    session.refresh(obj)

    return BucketObjectResponse(
        id=obj.id,
        bucket_id=bucket.id,
        name=obj.name,
        size=obj.size,
        content_type=obj.content_type,
        created_at=obj.created_at.isoformat(),
    )


@router.get(
    "/{bucket_id}/objects",
    response_model=ObjectListResponse,
    summary="List objects in bucket",
)
async def list_bucket_objects(
    bucket_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> ObjectListResponse:
    """List all objects stored in a bucket."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:read"):
        raise HTTPException(status_code=403, detail="Missing storage:read permission.")

    user_id = _resolve_user_id(caller)
    bucket = session.get(BucketRecord, bucket_id)
    if not bucket or bucket.user_id != user_id:
        bucket = session.exec(
            select(BucketRecord).where(
                BucketRecord.user_id == user_id,
                BucketRecord.name == bucket_id,
            )
        ).first()

    if not bucket or bucket.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_id}' not found.")

    objs = session.exec(
        select(BucketObjectRecord)
        .where(BucketObjectRecord.bucket_id == bucket.id)
        .order_by(BucketObjectRecord.created_at.desc())
    ).all()

    data = [
        BucketObjectResponse(
            id=o.id,
            bucket_id=bucket.id,
            name=o.name,
            size=o.size,
            content_type=o.content_type,
            created_at=o.created_at.isoformat(),
        )
        for o in objs
    ]
    return ObjectListResponse(data=data)


@router.get(
    "/{bucket_id}/objects/{object_name:path}",
    summary="Download object",
)
async def download_bucket_object(
    bucket_id: str,
    object_name: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    """Download the raw bytes of an object."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:read"):
        raise HTTPException(status_code=403, detail="Missing storage:read permission.")

    user_id = _resolve_user_id(caller)
    bucket = session.get(BucketRecord, bucket_id)
    if not bucket or bucket.user_id != user_id:
        bucket = session.exec(
            select(BucketRecord).where(
                BucketRecord.user_id == user_id,
                BucketRecord.name == bucket_id,
            )
        ).first()

    if not bucket or bucket.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_id}' not found.")

    obj = session.exec(
        select(BucketObjectRecord).where(
            BucketObjectRecord.bucket_id == bucket.id,
            (BucketObjectRecord.name == object_name) | (BucketObjectRecord.id == object_name),
        )
    ).first()

    if not obj:
        raise HTTPException(status_code=404, detail=f"Object '{object_name}' not found.")

    return Response(
        content=obj.content,
        media_type=obj.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{obj.name.split("/")[-1]}"',
            "Content-Length": str(obj.size),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/{bucket_id}/objects/{object_name:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete object",
)
async def delete_bucket_object(
    bucket_id: str,
    object_name: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
):
    """Delete an object from a bucket."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "storage:write"):
        raise HTTPException(status_code=403, detail="Missing storage:write permission.")

    user_id = _resolve_user_id(caller)
    bucket = session.get(BucketRecord, bucket_id)
    if not bucket or bucket.user_id != user_id:
        bucket = session.exec(
            select(BucketRecord).where(
                BucketRecord.user_id == user_id,
                BucketRecord.name == bucket_id,
            )
        ).first()

    if not bucket or bucket.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Bucket '{bucket_id}' not found.")

    obj = session.exec(
        select(BucketObjectRecord).where(
            BucketObjectRecord.bucket_id == bucket.id,
            (BucketObjectRecord.name == object_name) | (BucketObjectRecord.id == object_name),
        )
    ).first()

    if not obj:
        raise HTTPException(status_code=404, detail=f"Object '{object_name}' not found.")

    session.delete(obj)
    session.commit()
    return None
