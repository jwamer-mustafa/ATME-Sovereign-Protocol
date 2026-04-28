"""Tests for the FastAPI gateway."""

import pytest
from httpx import ASGITransport, AsyncClient

from backend.gateway.app import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_get_tiers():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/billing/tiers")
        assert response.status_code == 200
        data = response.json()
        assert "tiers" in data
        tier_ids = [t["id"] for t in data["tiers"]]
        assert "free" in tier_ids
        assert "pro" in tier_ids
        assert "creator" in tier_ids


@pytest.mark.asyncio
async def test_register_and_login():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register
        response = await client.post("/api/auth/register", json={
            "email": "test@example.com",
            "password": "testpass123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["tier"] == "free"

        # Login
        response = await client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "testpass123",
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data


@pytest.mark.asyncio
async def test_register_duplicate():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/auth/register", json={
            "email": "dup@example.com",
            "password": "pass",
        })
        response = await client.post("/api/auth/register", json={
            "email": "dup@example.com",
            "password": "pass",
        })
        assert response.status_code == 400


@pytest.mark.asyncio
async def test_unauthenticated_inject():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/simulation/event", json={"text": "hello"})
        assert response.status_code == 401
