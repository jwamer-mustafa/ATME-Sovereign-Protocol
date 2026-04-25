"""
Stripe Billing integration with tier management.
Tiers:
  FREE: Watch only (5-10s delay), no events
  PRO: Live stream + 5 events/day
  CREATOR: Full control, unlimited events
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class TierConfig:
    name: str
    price_monthly: float
    stream_delay: float  # seconds
    events_per_day: int
    live_stream: bool
    env_control: bool
    speed_control: bool


TIERS: dict[str, TierConfig] = {
    "free": TierConfig(
        name="Free",
        price_monthly=0,
        stream_delay=5.0,
        events_per_day=0,
        live_stream=False,
        env_control=False,
        speed_control=False,
    ),
    "pro": TierConfig(
        name="Pro",
        price_monthly=9.99,
        stream_delay=0,
        events_per_day=5,
        live_stream=True,
        env_control=False,
        speed_control=True,
    ),
    "creator": TierConfig(
        name="Creator",
        price_monthly=29.99,
        stream_delay=0,
        events_per_day=999999,
        live_stream=True,
        env_control=True,
        speed_control=True,
    ),
}


class BillingManager:
    """Manages user billing state and tier enforcement."""

    def __init__(self):
        self.user_events: dict[str, dict[str, Any]] = {}

    def get_tier_config(self, tier: str) -> TierConfig:
        return TIERS.get(tier, TIERS["free"])

    def check_event_limit(self, user_id: str, tier: str) -> bool:
        """Check if user can inject an event based on their tier."""
        config = self.get_tier_config(tier)
        today = time.strftime("%Y-%m-%d")

        user_data = self.user_events.get(user_id, {})
        if user_data.get("date") != today:
            user_data = {"date": today, "count": 0}
            self.user_events[user_id] = user_data

        return user_data["count"] < config.events_per_day

    def record_event(self, user_id: str) -> None:
        today = time.strftime("%Y-%m-%d")
        user_data = self.user_events.get(user_id, {"date": today, "count": 0})
        if user_data.get("date") != today:
            user_data = {"date": today, "count": 0}
        user_data["count"] += 1
        self.user_events[user_id] = user_data

    def get_remaining_events(self, user_id: str, tier: str) -> int:
        config = self.get_tier_config(tier)
        today = time.strftime("%Y-%m-%d")
        user_data = self.user_events.get(user_id, {"date": today, "count": 0})
        if user_data.get("date") != today:
            return config.events_per_day
        return max(0, config.events_per_day - user_data["count"])

    def can_stream_live(self, tier: str) -> bool:
        return self.get_tier_config(tier).live_stream

    def can_control_env(self, tier: str) -> bool:
        return self.get_tier_config(tier).env_control

    def can_control_speed(self, tier: str) -> bool:
        return self.get_tier_config(tier).speed_control

    def get_stream_delay(self, tier: str) -> float:
        return self.get_tier_config(tier).stream_delay

    def get_all_tiers(self) -> list[dict[str, Any]]:
        return [
            {
                "id": tid,
                "name": tc.name,
                "price_monthly": tc.price_monthly,
                "stream_delay": tc.stream_delay,
                "events_per_day": tc.events_per_day,
                "live_stream": tc.live_stream,
                "env_control": tc.env_control,
                "speed_control": tc.speed_control,
            }
            for tid, tc in TIERS.items()
        ]


billing_manager = BillingManager()
