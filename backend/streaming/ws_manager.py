"""
WebSocket streaming manager.
Two channels:
1) State channel (60 FPS): position, velocity, actions
2) Perception channel (10-20 FPS): frame, attention, metrics
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image


class ConnectionManager:
    """Manages WebSocket connections with tier-based access control."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.connection_tiers: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str,
                      tier: str = "free") -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections[client_id] = websocket
            self.connection_tiers[client_id] = tier

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            self.active_connections.pop(client_id, None)
            self.connection_tiers.pop(client_id, None)

    async def broadcast_state(self, state: dict[str, Any]) -> None:
        """Broadcast environment state to all connected clients."""
        message = json.dumps({"type": "state", "data": state, "ts": time.time()})
        await self._broadcast(message)

    async def broadcast_perception(self, perception_data: dict[str, Any]) -> None:
        """Broadcast perception data (frames, attention maps, metrics)."""
        message = json.dumps({
            "type": "perception",
            "data": perception_data,
            "ts": time.time(),
        })
        await self._broadcast(message, perception=True)

    async def send_event_response(self, client_id: str,
                                  response: dict[str, Any]) -> None:
        ws = self.active_connections.get(client_id)
        if ws:
            try:
                await ws.send_json({
                    "type": "event_response",
                    "data": response,
                    "ts": time.time(),
                })
            except WebSocketDisconnect:
                await self.disconnect(client_id)

    async def _broadcast(self, message: str, perception: bool = False) -> None:
        disconnected = []
        async with self._lock:
            for client_id, ws in self.active_connections.items():
                tier = self.connection_tiers.get(client_id, "free")
                if tier == "free" and perception:
                    continue
                try:
                    await ws.send_text(message)
                except (WebSocketDisconnect, RuntimeError):
                    disconnected.append(client_id)

        for client_id in disconnected:
            await self.disconnect(client_id)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


def encode_frame_base64(frame: np.ndarray, quality: int = 70) -> str:
    """Encode RGB numpy array to base64 JPEG string."""
    img = Image.fromarray(frame.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_heatmap_base64(attention_map: np.ndarray, quality: int = 60) -> str:
    """Encode attention/saliency heatmap to base64 JPEG."""
    if attention_map.ndim == 4:
        attention_map = attention_map[0, 0]
    elif attention_map.ndim == 3:
        attention_map = attention_map[0]

    heatmap = (attention_map * 255).astype(np.uint8)
    img = Image.fromarray(heatmap, mode="L")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def prepare_perception_payload(
    frame: np.ndarray,
    attention_map: np.ndarray,
    action_info: dict[str, Any],
    features: np.ndarray | None = None,
) -> dict[str, Any]:
    """Prepare compressed perception payload for streaming."""
    payload: dict[str, Any] = {
        "frame_raw": encode_frame_base64(frame),
        "attention_map": encode_heatmap_base64(attention_map),
        "action": action_info.get("action", [0, 0, 0]),
        "confidence": action_info.get("confidence", 0.0),
        "entropy": action_info.get("entropy", 0.0),
        "action_logits": action_info.get("action_mean", [0, 0, 0]),
    }

    if features is not None:
        if isinstance(features, np.ndarray):
            payload["features"] = features.tolist()

    return payload
