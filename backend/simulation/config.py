"""Global configuration for the platform."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Embodied AI Platform"
    debug: bool = True

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/embodied_ai"
    redis_url: str = "redis://localhost:6379"

    # Auth
    jwt_secret: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Stripe
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # Simulation
    render_width: int = 128
    render_height: int = 128
    sim_fps: int = 60
    perception_fps: int = 15

    # Tiers
    free_stream_delay: float = 5.0
    free_events_per_day: int = 0
    pro_events_per_day: int = 5
    creator_events_per_day: int = 999999

    model_config = {"env_prefix": "EAI_", "env_file": ".env"}


settings = Settings()
