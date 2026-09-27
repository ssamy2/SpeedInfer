"""SpeedInfer API Gateway Package.

Provides high-throughput OpenAI-compatible API routing, rate limiting,
metering enforcement, and backend load balancing.
"""

from speedinfer.gateway.app import app, create_app, get_proxy, get_registry

__all__ = ["app", "create_app", "get_proxy", "get_registry"]
