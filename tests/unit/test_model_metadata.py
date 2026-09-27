"""Ensure console model cards get configured metadata rather than fixed prices."""

import pytest

from speedinfer.database.models import ModelVersion, User
from speedinfer.engine.registry import ModelRegistry
from speedinfer.gateway.routes.models import get_model, list_models


@pytest.mark.asyncio
async def test_registered_model_metadata(db_session):
    registry = ModelRegistry()
    registry.register_model(
        name="example/custom-model",
        context_length=8192,
        prompt_price_per_million=1.25,
        completion_price_per_million=2.5,
    )
    caller = User(email="developer@example.test")
    response = await list_models(caller, registry, db_session)
    model = response.data[0]
    assert (
        model.context_length,
        model.prompt_price_per_million,
        model.completion_price_per_million,
    ) == (8192, 1.25, 2.5)
    detail = await get_model("example/custom-model", caller, registry, db_session)
    assert detail.context_length == 8192 and detail.prompt_price_per_million == 1.25


@pytest.mark.asyncio
async def test_database_only_model_metadata(db_session):
    db_session.add(
        ModelVersion(
            name="example/database-model",
            base_model_path="example/base",
            context_length=4096,
            prompt_price_per_million=0.75,
            completion_price_per_million=1.5,
        )
    )
    db_session.commit()
    caller = User(email="developer@example.test")
    response = await list_models(caller, ModelRegistry(), db_session)
    model = response.data[0]
    assert (
        model.context_length,
        model.prompt_price_per_million,
        model.completion_price_per_million,
    ) == (4096, 0.75, 1.5)
    detail = await get_model("example/database-model", caller, ModelRegistry(), db_session)
    assert detail.completion_price_per_million == 1.5
