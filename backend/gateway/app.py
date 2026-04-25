"""
FastAPI Gateway — Auth, Billing, Session Management, REST + WebSocket endpoints.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from backend.gateway.auth import (
    User,
    create_access_token,
    create_refresh_token,
    decode_token,
    user_store,
)
from backend.gateway.billing import billing_manager
from backend.simulation.config import settings
from backend.simulation.orchestrator import SimulationOrchestrator

orchestrator: SimulationOrchestrator | None = None
sim_task: asyncio.Task | None = None
security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator, sim_task
    orchestrator = SimulationOrchestrator()
    sim_task = asyncio.create_task(orchestrator.run_loop(
        target_fps=settings.sim_fps,
        perception_fps=settings.perception_fps,
    ))
    yield
    if orchestrator:
        orchestrator.stop()
    if sim_task:
        sim_task.cancel()
        try:
            await sim_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Auth models ---

class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    tier: str


class EventRequest(BaseModel):
    text: str


class SpeedRequest(BaseModel):
    multiplier: float


class EnvParamRequest(BaseModel):
    param: str
    value: float


# --- Dependencies ---

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User | None:
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload:
        return None
    return user_store.get_by_id(payload.get("sub", ""))


async def require_user(
    user: User | None = Depends(get_current_user),
) -> User:
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


async def require_tier(
    min_tier: str,
    user: User = Depends(require_user),
) -> User:
    tier_levels = {"free": 0, "pro": 1, "creator": 2}
    if tier_levels.get(user.tier, 0) < tier_levels.get(min_tier, 0):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires {min_tier} tier or higher",
        )
    return user


# --- Auth endpoints ---

@app.post("/api/auth/register", response_model=TokenResponse)
async def register(req: RegisterRequest):
    existing = user_store.get_by_email(req.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = user_store.create_user(req.email, req.password)
    access = create_access_token(user.user_id, user.email, user.tier)
    refresh = create_refresh_token(user.user_id)

    return TokenResponse(
        access_token=access, refresh_token=refresh, tier=user.tier,
    )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    user = user_store.get_by_email(req.email)
    if not user or not user_store.verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = create_access_token(user.user_id, user.email, user.tier)
    refresh = create_refresh_token(user.user_id)

    return TokenResponse(
        access_token=access, refresh_token=refresh, tier=user.tier,
    )


@app.get("/api/auth/me")
async def get_me(user: User = Depends(require_user)):
    return {
        "user_id": user.user_id,
        "email": user.email,
        "tier": user.tier,
        "events_remaining": billing_manager.get_remaining_events(user.user_id, user.tier),
    }


# --- Billing endpoints ---

@app.get("/api/billing/tiers")
async def get_tiers():
    return {"tiers": billing_manager.get_all_tiers()}


@app.post("/api/billing/upgrade")
async def upgrade_tier(tier: str, user: User = Depends(require_user)):
    if tier not in ("pro", "creator"):
        raise HTTPException(status_code=400, detail="Invalid tier")
    user_store.update_tier(user.user_id, tier)
    access = create_access_token(user.user_id, user.email, tier)
    return {"status": "upgraded", "tier": tier, "access_token": access}


# --- Simulation endpoints ---

@app.get("/api/simulation/state")
async def get_state(user: User | None = Depends(get_current_user)):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Simulation not running")
    return orchestrator.get_full_state()


@app.post("/api/simulation/event")
async def inject_event(req: EventRequest, user: User = Depends(require_user)):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Simulation not running")

    if not billing_manager.check_event_limit(user.user_id, user.tier):
        raise HTTPException(
            status_code=429,
            detail=f"Event limit reached for {user.tier} tier",
        )

    result = orchestrator.inject_question(req.text, user_id=user.user_id)
    billing_manager.record_event(user.user_id)

    return {
        **result,
        "events_remaining": billing_manager.get_remaining_events(user.user_id, user.tier),
    }


@app.post("/api/simulation/speed")
async def set_speed(req: SpeedRequest, user: User = Depends(require_user)):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Simulation not running")

    if not billing_manager.can_control_speed(user.tier):
        raise HTTPException(status_code=403, detail="Speed control requires Pro tier or higher")

    orchestrator.set_speed(req.multiplier)
    return {"speed": orchestrator.speed_multiplier}


@app.post("/api/simulation/pause")
async def toggle_pause(user: User = Depends(require_user)):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Simulation not running")
    paused = orchestrator.toggle_pause()
    return {"paused": paused}


@app.get("/api/simulation/training")
async def get_training_stats():
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Simulation not running")
    return {
        "total_steps": orchestrator.trainer.total_steps,
        "episode": orchestrator.episode_count,
        "stats": orchestrator.trainer.training_stats[-10:],
    }


@app.get("/api/simulation/memory")
async def get_memory(user: User = Depends(require_user)):
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Simulation not running")
    return {
        "recent": orchestrator.memory.get_recent(20),
        "episodic_size": orchestrator.memory.size,
        "replay_size": orchestrator.replay_buffer.size,
    }


@app.get("/api/simulation/events")
async def get_events():
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Simulation not running")

    active = orchestrator.event_injector.get_active_events()
    resolved = orchestrator.event_injector.get_resolved_events()

    return {
        "active": [
            {"event_id": e.event_id, "text": e.text, "position": e.position}
            for e in active
        ],
        "resolved": [
            {
                "event_id": e.event_id,
                "text": e.text,
                "response": e.response,
                "confidence": e.confidence,
            }
            for e in resolved
        ],
    }


# --- WebSocket endpoint ---

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    if not orchestrator:
        await websocket.close(code=1011)
        return

    token = websocket.query_params.get("token", "")
    tier = "free"
    user_id = client_id

    if token:
        payload = decode_token(token)
        if payload:
            tier = payload.get("tier", "free")
            user_id = payload.get("sub", client_id)

    await orchestrator.ws_manager.connect(websocket, client_id, tier)

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "inject_event":
                if not billing_manager.check_event_limit(user_id, tier):
                    await websocket.send_json({
                        "type": "error",
                        "message": "Event limit reached",
                    })
                    continue

                result = orchestrator.inject_question(
                    data.get("text", ""), user_id=user_id,
                )
                billing_manager.record_event(user_id)
                await websocket.send_json({"type": "event_injected", "data": result})

            elif msg_type == "set_speed":
                if billing_manager.can_control_speed(tier):
                    orchestrator.set_speed(data.get("multiplier", 1.0))

            elif msg_type == "toggle_pause":
                paused = orchestrator.toggle_pause()
                await websocket.send_json({"type": "pause_state", "paused": paused})

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong", "ts": data.get("ts", 0)})

    except WebSocketDisconnect:
        await orchestrator.ws_manager.disconnect(client_id)


# --- Health check ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "simulation_running": orchestrator is not None and orchestrator.running,
        "connections": orchestrator.ws_manager.connection_count if orchestrator else 0,
    }
