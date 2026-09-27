"""OpenAI-compatible Files API endpoints.

Provides:
- POST /v1/files: Upload a file (e.g. for fine-tuning or storage).
- GET /v1/files: List user files.
- GET /v1/files/{file_id}: Retrieve file metadata.
- GET /v1/files/{file_id}/content: Download file content.
- DELETE /v1/files/{file_id}: Delete a file.
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session, select

from speedinfer.core.auth import check_scope_permission
from speedinfer.database.models import ApiKey, FileRecord, User
from speedinfer.database.session import get_session
from speedinfer.gateway.routes.models import get_caller_auth

router = APIRouter(prefix="/v1/files", tags=["Files"])

MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MiB limit per file


class FileObject(BaseModel):
    """OpenAI-compatible file object representation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    bytes: int
    created_at: int
    filename: str
    object: Literal["file"] = "file"
    purpose: str
    status: str = "uploaded"
    status_details: str | None = None


class FileListResponse(BaseModel):
    """OpenAI-compatible list response for files."""

    object: Literal["list"] = "list"
    data: list[FileObject]


class FileDeleteResponse(BaseModel):
    """OpenAI-compatible deletion status response."""

    id: str
    object: Literal["file"] = "file"
    deleted: bool


def _resolve_user_id(caller: ApiKey | User) -> int:
    return caller.user_id if isinstance(caller, ApiKey) else caller.id


@router.post(
    "",
    response_model=FileObject,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file",
    description=(
        "Upload a file that can be used across various endpoints (e.g. fine-tuning, storage)."
    ),
)
@router.post(
    "/",
    response_model=FileObject,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def upload_file(
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
    file: Annotated[
        UploadFile, File(description="The File object (not file name) to be uploaded.")
    ],
    purpose: Annotated[
        str, Form(description="The intended purpose of the uploaded file.")
    ] = "fine-tune",
) -> FileObject:
    """Upload a file to SpeedInfer storage."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "files:write"):
        raise HTTPException(status_code=403, detail="Missing files:write permission.")

    user_id = _resolve_user_id(caller)
    filename = file.filename or "uploaded_file"

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content cannot be empty.",
        )
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_BYTES // (1024 * 1024)} MiB.",
        )

    # Validate JSONL if purpose is fine-tune
    if purpose in {"fine-tune", "fine_tune", "training"} and filename.lower().endswith(".jsonl"):
        import json

        lines = content.decode("utf-8", errors="replace").splitlines()
        valid_lines = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError
                valid_lines += 1
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSONL format: each line must be a valid JSON object.",
                ) from None
        if valid_lines == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty dataset: JSONL contains no valid records.",
            )

    rec = FileRecord(
        user_id=user_id,
        filename=filename,
        size_bytes=len(content),
        purpose=purpose,
        status="uploaded",
        content=content,
    )
    session.add(rec)
    session.commit()
    session.refresh(rec)

    return FileObject(
        id=rec.id,
        bytes=rec.size_bytes,
        created_at=rec.created_at,
        filename=rec.filename,
        purpose=rec.purpose,
        status=rec.status,
    )


@router.get(
    "",
    response_model=FileListResponse,
    summary="List files",
    description="Returns a list of files that belong to the user's organization.",
)
@router.get(
    "/",
    response_model=FileListResponse,
    include_in_schema=False,
)
async def list_files(
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
    purpose: str | None = None,
) -> FileListResponse:
    """Retrieve metadata of all files owned by current user."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "files:read"):
        raise HTTPException(status_code=403, detail="Missing files:read permission.")

    user_id = _resolve_user_id(caller)
    stmt = select(FileRecord).where(FileRecord.user_id == user_id)
    if purpose:
        stmt = stmt.where(FileRecord.purpose == purpose)
    stmt = stmt.order_by(FileRecord.created_at.desc())

    records = session.exec(stmt).all()
    data = [
        FileObject(
            id=r.id,
            bytes=r.size_bytes,
            created_at=r.created_at,
            filename=r.filename,
            purpose=r.purpose,
            status=r.status,
        )
        for r in records
    ]
    return FileListResponse(object="list", data=data)


@router.get(
    "/{file_id}",
    response_model=FileObject,
    summary="Retrieve file",
    description="Returns information about a specific file.",
)
async def get_file(
    file_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> FileObject:
    """Retrieve metadata for a specific file ID."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "files:read"):
        raise HTTPException(status_code=403, detail="Missing files:read permission.")

    user_id = _resolve_user_id(caller)
    rec = session.get(FileRecord, file_id)
    if rec is None or rec.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found.",
        )

    return FileObject(
        id=rec.id,
        bytes=rec.size_bytes,
        created_at=rec.created_at,
        filename=rec.filename,
        purpose=rec.purpose,
        status=rec.status,
    )


@router.get(
    "/{file_id}/content",
    summary="Retrieve file content",
    description="Returns the contents of the specified file.",
)
async def get_file_content(
    file_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> Response:
    """Download raw binary content of a file."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "files:read"):
        raise HTTPException(status_code=403, detail="Missing files:read permission.")

    user_id = _resolve_user_id(caller)
    rec = session.get(FileRecord, file_id)
    if rec is None or rec.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found.",
        )

    return Response(
        content=rec.content,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{rec.filename}"',
            "Content-Length": str(rec.size_bytes),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete(
    "/{file_id}",
    response_model=FileDeleteResponse,
    summary="Delete a file",
    description="Delete a file.",
)
async def delete_file(
    file_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> FileDeleteResponse:
    """Delete a file belonging to the authenticated user."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "files:write"):
        raise HTTPException(status_code=403, detail="Missing files:write permission.")

    user_id = _resolve_user_id(caller)
    rec = session.get(FileRecord, file_id)
    if rec is None or rec.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{file_id}' not found.",
        )

    session.delete(rec)
    session.commit()

    return FileDeleteResponse(id=file_id, object="file", deleted=True)
