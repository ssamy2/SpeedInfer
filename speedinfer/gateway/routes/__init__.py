"""SpeedInfer Gateway API Routes Package."""

from speedinfer.gateway.routes.auth import router as auth_router
from speedinfer.gateway.routes.billing import router as billing_router
from speedinfer.gateway.routes.chat import router as chat_router
from speedinfer.gateway.routes.completions import router as completions_router
from speedinfer.gateway.routes.health import router as health_router
from speedinfer.gateway.routes.keys import router as keys_router
from speedinfer.gateway.routes.models import router as models_router
from speedinfer.gateway.routes.usage import router as usage_router

__all__ = [
    "auth_router",
    "billing_router",
    "chat_router",
    "completions_router",
    "health_router",
    "keys_router",
    "models_router",
    "usage_router",
]
