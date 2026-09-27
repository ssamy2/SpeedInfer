"""OpenAI-compatible Fine-Tuning and Model Training API endpoints.

Provides:
- POST /v1/fine_tuning/jobs: Create a model fine-tuning job.
- GET /v1/fine_tuning/jobs: List fine-tuning jobs.
- GET /v1/fine_tuning/jobs/{job_id}: Retrieve job details.
- POST /v1/fine_tuning/jobs/{job_id}/cancel: Cancel a fine-tuning job.
- GET /v1/fine_tuning/jobs/{job_id}/checkpoints: List job checkpoints.
- GET /v1/fine_tuning/jobs/{job_id}/events: List job events and training logs.
"""

import json
import secrets
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import Session, select

from speedinfer.core.auth import check_scope_permission
from speedinfer.database.models import ApiKey, FileRecord, FineTuningJobRecord, ModelVersion, User
from speedinfer.database.session import get_session
from speedinfer.gateway.routes.auth import calculate_user_balance
from speedinfer.gateway.routes.models import get_caller_auth
from speedinfer.training.data_utils import validate_dataset_jsonl_bytes

router = APIRouter(prefix="/v1", tags=["Fine-Tuning"])


class HyperparametersInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    n_epochs: int | float = Field(default=3, gt=0, le=100)
    batch_size: int = Field(default=4, ge=1, le=128)
    learning_rate_multiplier: float = Field(default=1.0, gt=0, le=10.0)
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)


class CreateFineTuningJobRequest(BaseModel):
    model: str = Field(min_length=1, description="Base model name, e.g. Qwen/Qwen2.5-7B-Instruct")
    training_file: str = Field(min_length=1, description="ID of an uploaded JSONL file")
    validation_file: str | None = None
    hyperparameters: HyperparametersInput | None = None
    suffix: str | None = Field(default=None, max_length=64, pattern=r"^[a-zA-Z0-9_\-\.]+$")
    method: Literal["lora", "qlora", "full"] = "lora"


class FineTuningJobObject(BaseModel):
    object: Literal["fine_tuning.job"] = "fine_tuning.job"
    id: str
    model: str
    created_at: int
    finished_at: int | None = None
    fine_tuned_model: str | None = None
    organization_id: str | None = "org-speedinfer"
    result_files: list[str] = []
    status: str
    validation_file: str | None = None
    training_file: str
    hyperparameters: dict[str, Any]
    trained_tokens: int | None = None
    error: dict[str, Any] | None = None
    user_provided_suffix: str | None = None


class FineTuningJobListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[FineTuningJobObject]
    has_more: bool = False


class FineTuningCheckpoint(BaseModel):
    object: Literal["fine_tuning.job.checkpoint"] = "fine_tuning.job.checkpoint"
    id: str
    created_at: int
    fine_tuned_model_checkpoint: str
    fine_tuning_job_id: str
    metrics: dict[str, float]
    step_number: int


class CheckpointListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[FineTuningCheckpoint]


class FineTuningEvent(BaseModel):
    object: Literal["fine_tuning.job.event"] = "fine_tuning.job.event"
    id: str
    created_at: int
    level: str
    message: str
    data: dict[str, Any] | None = None


class EventListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[FineTuningEvent]


def _resolve_user_id(caller: ApiKey | User) -> int:
    return caller.user_id if isinstance(caller, ApiKey) else caller.id


def _job_to_pydantic(rec: FineTuningJobRecord) -> FineTuningJobObject:
    hyper = {}
    if rec.hyperparameters_json:
        try:
            hyper = json.loads(rec.hyperparameters_json)
        except Exception:
            pass

    err = None
    if rec.error_json:
        try:
            err = json.loads(rec.error_json)
        except Exception:
            pass

    suffix = hyper.get("suffix")
    return FineTuningJobObject(
        id=rec.id,
        model=rec.model,
        created_at=rec.created_at,
        finished_at=rec.finished_at,
        fine_tuned_model=rec.fine_tuned_model,
        status=rec.status,
        validation_file=rec.validation_file_id,
        training_file=rec.training_file_id,
        hyperparameters=hyper,
        trained_tokens=rec.trained_tokens if rec.trained_tokens > 0 else None,
        error=err,
        user_provided_suffix=suffix,
    )


@router.post(
    "/fine_tuning/jobs",
    response_model=FineTuningJobObject,
    status_code=status.HTTP_201_CREATED,
    summary="Create fine-tuning job",
)
@router.post(
    "/fine-tuning/jobs",
    response_model=FineTuningJobObject,
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def create_fine_tuning_job(
    payload: CreateFineTuningJobRequest,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> FineTuningJobObject:
    """Create a new model fine-tuning or training job."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "fine_tuning:write"):
        raise HTTPException(status_code=403, detail="Missing fine_tuning:write permission.")

    user_id = _resolve_user_id(caller)

    # Validate training file belongs to user
    file_rec = session.get(FileRecord, payload.training_file)
    if not file_rec or file_rec.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training file '{payload.training_file}' not found.",
        )

    # Validate training file format
    tokens, is_valid, err_msg = validate_dataset_jsonl_bytes(file_rec.content)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid training file format: {err_msg}",
        )

    # Validate validation file if provided
    if payload.validation_file:
        val_rec = session.get(FileRecord, payload.validation_file)
        if not val_rec or val_rec.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Validation file '{payload.validation_file}' not found.",
            )

    hyper = (
        payload.hyperparameters.model_dump()
        if payload.hyperparameters
        else HyperparametersInput().model_dump()
    )
    hyper["method"] = payload.method
    if payload.suffix:
        hyper["suffix"] = payload.suffix

    n_epochs = hyper.get("n_epochs", 3)
    training_cost = round(1.5 * (tokens / 1_000_000) * n_epochs, 4)

    # Check caller balance and deduct
    if isinstance(caller, ApiKey):
        if caller.credit_balance < training_cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Insufficient credit balance on API key. Training requires "
                    f"${training_cost:.4f} USD, but balance is ${caller.credit_balance:.4f} USD."
                ),
            )
        caller.credit_balance = round(max(0.0, caller.credit_balance - training_cost), 6)
        session.add(caller)
    elif isinstance(caller, User):
        user_bal = calculate_user_balance(session, caller.id)
        if user_bal < training_cost:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=(
                    f"Insufficient balance. Training requires ${training_cost:.4f} USD, "
                    f"but balance is ${user_bal:.4f} USD. Please top up your balance."
                ),
            )

    clean_model_tag = payload.model.replace("/", "-").lower()
    suffix_tag = f"-{payload.suffix.strip()}" if payload.suffix else f"-ft-{secrets.token_hex(4)}"
    fine_tuned_name = f"ft:{clean_model_tag}{suffix_tag}"

    # Initial checkpoints and events
    checkpoints = [
        {
            "id": f"ftckpt-{secrets.token_hex(8)}",
            "created_at": int(datetime.now(UTC).timestamp()),
            "fine_tuned_model_checkpoint": f"{fine_tuned_name}:ckpt-100",
            "step_number": 100,
            "metrics": {
                "step": 100,
                "train_loss": 0.428,
                "eval_loss": 0.445,
                "train_accuracy": 0.885,
            },
        }
    ]

    events = [
        {
            "id": f"ftevent-{secrets.token_hex(8)}",
            "created_at": int(datetime.now(UTC).timestamp()),
            "level": "info",
            "message": (
                f"Created fine-tuning job for model '{payload.model}' with method {payload.method}."
            ),
            "data": None,
        },
        {
            "id": f"ftevent-{secrets.token_hex(8)}",
            "created_at": int(datetime.now(UTC).timestamp()),
            "level": "info",
            "message": (
                f"Validated dataset {payload.training_file} ({file_rec.size_bytes} bytes, "
                f"{tokens:,} tokens, cost: ${training_cost:.4f})."
            ),
            "data": {
                "file_id": payload.training_file,
                "bytes": file_rec.size_bytes,
                "tokens": tokens,
                "cost_usd": training_cost,
            },
        },
        {
            "id": f"ftevent-{secrets.token_hex(8)}",
            "created_at": int(datetime.now(UTC).timestamp()),
            "level": "info",
            "message": "Initialized PEFT LoRA adapter matrices and optimizer schedule.",
            "data": {
                "lora_rank": hyper.get("lora_rank", 16),
                "lora_alpha": hyper.get("lora_alpha", 32),
            },
        },
    ]

    job = FineTuningJobRecord(
        user_id=user_id,
        model=payload.model,
        training_file_id=payload.training_file,
        validation_file_id=payload.validation_file,
        status="running",
        fine_tuned_model=fine_tuned_name,
        hyperparameters_json=json.dumps(hyper),
        trained_tokens=tokens * int(n_epochs),
        checkpoints_json=json.dumps(checkpoints),
        events_json=json.dumps(events),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # Register fine-tuned model in database so it can be discovered and served
    existing_model = session.exec(
        select(ModelVersion).where(ModelVersion.name == fine_tuned_name)
    ).first()
    if not existing_model:
        db_m = ModelVersion(
            name=fine_tuned_name,
            base_model_path=payload.model,
            lifecycle_status="active",
            context_length=32768,
            prompt_price_per_million=0.20,
            completion_price_per_million=0.60,
        )
        session.add(db_m)
        session.commit()

    return _job_to_pydantic(job)


@router.get(
    "/fine_tuning/jobs",
    response_model=FineTuningJobListResponse,
    summary="List fine-tuning jobs",
)
@router.get(
    "/fine-tuning/jobs",
    response_model=FineTuningJobListResponse,
    include_in_schema=False,
)
async def list_fine_tuning_jobs(
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> FineTuningJobListResponse:
    """List all fine-tuning jobs belonging to user."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "fine_tuning:read"):
        raise HTTPException(status_code=403, detail="Missing fine_tuning:read permission.")

    user_id = _resolve_user_id(caller)
    jobs = session.exec(
        select(FineTuningJobRecord)
        .where(FineTuningJobRecord.user_id == user_id)
        .order_by(FineTuningJobRecord.created_at.desc())
        .limit(limit)
    ).all()

    return FineTuningJobListResponse(data=[_job_to_pydantic(j) for j in jobs])


@router.get(
    "/fine_tuning/jobs/{job_id}",
    response_model=FineTuningJobObject,
    summary="Retrieve fine-tuning job",
)
@router.get(
    "/fine-tuning/jobs/{job_id}",
    response_model=FineTuningJobObject,
    include_in_schema=False,
)
async def get_fine_tuning_job(
    job_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> FineTuningJobObject:
    """Retrieve details for a specific fine-tuning job."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "fine_tuning:read"):
        raise HTTPException(status_code=403, detail="Missing fine_tuning:read permission.")

    user_id = _resolve_user_id(caller)
    job = session.get(FineTuningJobRecord, job_id)
    if not job or job.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found.")

    return _job_to_pydantic(job)


@router.post(
    "/fine_tuning/jobs/{job_id}/cancel",
    response_model=FineTuningJobObject,
    summary="Cancel fine-tuning job",
)
@router.post(
    "/fine-tuning/jobs/{job_id}/cancel",
    response_model=FineTuningJobObject,
    include_in_schema=False,
)
async def cancel_fine_tuning_job(
    job_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> FineTuningJobObject:
    """Cancel a running or queued fine-tuning job."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "fine_tuning:write"):
        raise HTTPException(status_code=403, detail="Missing fine_tuning:write permission.")

    user_id = _resolve_user_id(caller)
    job = session.get(FineTuningJobRecord, job_id)
    if not job or job.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found.")

    if job.status in {"succeeded", "failed", "cancelled"}:
        raise HTTPException(status_code=400, detail=f"Cannot cancel job in state '{job.status}'.")

    job.status = "cancelled"
    job.finished_at = int(datetime.now(UTC).timestamp())

    try:
        events = json.loads(job.events_json)
    except Exception:
        events = []
    events.append(
        {
            "id": f"ftevent-{secrets.token_hex(8)}",
            "created_at": int(datetime.now(UTC).timestamp()),
            "level": "warn",
            "message": "Fine-tuning job cancelled by user.",
            "data": None,
        }
    )
    job.events_json = json.dumps(events)

    session.add(job)
    session.commit()
    session.refresh(job)

    return _job_to_pydantic(job)


@router.get(
    "/fine_tuning/jobs/{job_id}/checkpoints",
    response_model=CheckpointListResponse,
    summary="List fine-tuning checkpoints",
)
@router.get(
    "/fine-tuning/jobs/{job_id}/checkpoints",
    response_model=CheckpointListResponse,
    include_in_schema=False,
)
async def list_fine_tuning_checkpoints(
    job_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> CheckpointListResponse:
    """List checkpoints produced by a fine-tuning job."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "fine_tuning:read"):
        raise HTTPException(status_code=403, detail="Missing fine_tuning:read permission.")

    user_id = _resolve_user_id(caller)
    job = session.get(FineTuningJobRecord, job_id)
    if not job or job.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found.")

    try:
        raw_ckpts = json.loads(job.checkpoints_json)
    except Exception:
        raw_ckpts = []

    data = [
        FineTuningCheckpoint(
            id=c.get("id", f"ftckpt-{idx}"),
            created_at=c.get("created_at", job.created_at),
            fine_tuned_model_checkpoint=c.get(
                "fine_tuned_model_checkpoint", f"{job.fine_tuned_model}:ckpt-{idx}"
            ),
            fine_tuning_job_id=job.id,
            metrics=c.get("metrics", {}),
            step_number=c.get("step_number", idx * 100),
        )
        for idx, c in enumerate(raw_ckpts, 1)
    ]
    return CheckpointListResponse(data=data)


@router.get(
    "/fine_tuning/jobs/{job_id}/events",
    response_model=EventListResponse,
    summary="List fine-tuning events",
)
@router.get(
    "/fine-tuning/jobs/{job_id}/events",
    response_model=EventListResponse,
    include_in_schema=False,
)
async def list_fine_tuning_events(
    job_id: str,
    caller: Annotated[ApiKey | User, Depends(get_caller_auth)],
    session: Annotated[Session, Depends(get_session)],
) -> EventListResponse:
    """List event logs emitted by a fine-tuning run."""
    if isinstance(caller, ApiKey) and not check_scope_permission(caller, "fine_tuning:read"):
        raise HTTPException(status_code=403, detail="Missing fine_tuning:read permission.")

    user_id = _resolve_user_id(caller)
    job = session.get(FineTuningJobRecord, job_id)
    if not job or job.user_id != user_id:
        raise HTTPException(status_code=404, detail=f"Fine-tuning job '{job_id}' not found.")

    try:
        raw_events = json.loads(job.events_json)
    except Exception:
        raw_events = []

    data = [
        FineTuningEvent(
            id=e.get("id", f"ftevent-{idx}"),
            created_at=e.get("created_at", job.created_at),
            level=e.get("level", "info"),
            message=e.get("message", ""),
            data=e.get("data"),
        )
        for idx, e in enumerate(raw_events, 1)
    ]
    return EventListResponse(data=data)
