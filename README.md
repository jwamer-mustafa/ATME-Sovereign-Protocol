# Embodied AI Platform

A real-time embodied AI platform featuring a 3D physics environment with an RL agent that learns, perceives, and interacts — streamed live to users via a web interface.

## Architecture

```
[Frontend React/Three.js]
  ├─ Scene (WebGL 3D)
  ├─ Perception Panels (Camera/Attention/Features/Decision)
  ├─ Interaction (Inject Event / Ask)
  └─ Billing UI

        ⇅ WebSocket (60 FPS state, 10–20 FPS perception)
        ⇅ REST (auth, billing, configs)

[Gateway - FastAPI]
  ├─ Auth (JWT)
  ├─ Billing (Stripe)
  ├─ Session Manager
  └─ Rate Limits / Tier Control

[Simulation Core - Python]
  ├─ Physics Env (PyBullet)
  ├─ Sensor Stack (RGB Camera)
  ├─ Agent (Policy + Value Network)
  ├─ Memory (Replay + Episodic)
  └─ Event Injector (Questions → World Entities)

[Learning Core]
  ├─ RL Algorithm (PPO)
  ├─ Vision Encoder (CNN)
  ├─ Attention/Saliency Export (Grad-CAM)
  └─ Checkpointing

[Streaming]
  ├─ Frame Encoder (JPEG/base64)
  ├─ State/Telemetry (WebSocket)
  └─ Perception Payload (compressed)
```

## Features

- **3D Physics Environment**: Continuous space with gravity, collisions, objects, and targets (PyBullet)
- **Embodied Agent**: Capsule agent with forward-facing camera, learns via PPO
- **Full Perception Pipeline**: Camera → CNN Encoder → Latent → Attention/Saliency → UI
- **Episodic Memory**: Embedding-based memory with cosine similarity retrieval
- **Event Injection**: Questions become physical world entities (query orbs)
- **Real-time Streaming**: WebSocket state (60 FPS) + perception (10-20 FPS)
- **Ethical Safeguards**: No consciousness claims, mandatory confidence display
- **SaaS Ready**: JWT auth, Stripe billing, tier-based access (Free/Pro/Creator)

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (for production)

### Development Setup

```bash
# Backend
pip install -e ".[dev]"
uvicorn backend.gateway.app:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
docker compose up --build
```

### Running Tests

```bash
pytest tests/ -v
```

## Subscription Tiers

| Feature | Free | Pro ($9.99/mo) | Creator ($29.99/mo) |
|---------|------|-----------------|---------------------|
| Stream | 5s delay | Live | Live |
| Events/day | 0 | 5 | Unlimited |
| Speed Control | No | Yes | Yes |
| Env Control | No | No | Yes |

## Ethical Guidelines

- No anthropomorphic claims ("I feel", "I want", etc.)
- Mandatory confidence display on all responses
- Transparent uncertainty reporting
- All responses include a computational disclaimer

## API Endpoints

### Auth
- `POST /api/auth/register` — Register new user
- `POST /api/auth/login` — Login
- `GET /api/auth/me` — Get current user info

### Billing
- `GET /api/billing/tiers` — List available tiers
- `POST /api/billing/upgrade` — Upgrade subscription

### Simulation
- `GET /api/simulation/state` — Get full simulation state
- `POST /api/simulation/event` — Inject question event
- `POST /api/simulation/speed` — Set simulation speed
- `POST /api/simulation/pause` — Toggle pause
- `GET /api/simulation/training` — Get training stats
- `GET /api/simulation/memory` — Get memory state
- `GET /api/simulation/events` — List active/resolved events

### WebSocket
- `ws://host/ws/{client_id}` — Real-time state + perception stream

## KPIs

- Latency < 120ms
- Session time > 6 minutes
- Day-1 Retention > 20%
- Conversion > 3%
