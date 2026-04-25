"""JWT Authentication module."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import jwt
from passlib.context import CryptContext

from backend.simulation.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass
class User:
    user_id: str
    email: str
    hashed_password: str
    tier: str = "free"  # free, pro, creator
    created_at: float = field(default_factory=time.time)
    events_today: int = 0
    events_reset_date: str = ""
    stripe_customer_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class InMemoryUserStore:
    """Simple in-memory user store (replace with Postgres in production)."""

    def __init__(self):
        self.users: dict[str, User] = {}
        self.email_index: dict[str, str] = {}

    def create_user(self, email: str, password: str, tier: str = "free") -> User:
        user_id = hashlib.sha256(f"{email}{time.time()}".encode()).hexdigest()[:16]
        hashed = pwd_context.hash(password)
        user = User(user_id=user_id, email=email, hashed_password=hashed, tier=tier)
        self.users[user_id] = user
        self.email_index[email] = user_id
        return user

    def get_by_email(self, email: str) -> User | None:
        uid = self.email_index.get(email)
        return self.users.get(uid) if uid else None

    def get_by_id(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def update_tier(self, user_id: str, tier: str) -> User | None:
        user = self.users.get(user_id)
        if user:
            user.tier = tier
        return user

    def verify_password(self, plain: str, hashed: str) -> bool:
        return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str, email: str, tier: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "tier": tier,
        "exp": time.time() + settings.jwt_expire_minutes * 60,
        "iat": time.time(),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_refresh_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": time.time() + 86400 * 30,
        "iat": time.time(),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except jwt.PyJWTError:
        return None


user_store = InMemoryUserStore()
