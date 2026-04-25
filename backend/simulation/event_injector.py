"""
Event Injection: Transform user questions/interactions into world entities.
Questions become physical query orbs — glowing, pulsing entities with 3D text
that the agent can approach, examine, or ignore.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OrbVisuals:
    """Visual properties for enhanced query orbs."""
    base_color: list[float] = field(default_factory=lambda: [0.2, 0.6, 1.0, 0.8])
    glow_color: list[float] = field(default_factory=lambda: [0.4, 0.8, 1.0, 0.6])
    pulse_speed: float = 2.0
    pulse_amplitude: float = 0.15
    glow_radius: float = 1.5
    text_3d: str = ""
    text_scale: float = 0.3
    particle_count: int = 8


@dataclass
class WorldEvent:
    event_id: str
    event_type: str  # "query_orb", "stimulus", "reward_zone"
    text: str
    position: list[float]
    body_id: int = -1
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    response: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    visuals: OrbVisuals = field(default_factory=OrbVisuals)
    reward_modifier: float = 0.0
    triggers_memory_recall: bool = False
    creates_new_goal: bool = False
    embedding: list[float] | None = None


class EventInjector:
    """Manages the lifecycle of injected events in the simulation."""

    def __init__(self):
        self.active_events: dict[str, WorldEvent] = {}
        self._event_counter = 0

    def create_question_event(self, text: str,
                              position: list[float] | None = None) -> WorldEvent:
        """
        Create a question event as an enhanced query_orb:
        glowing/pulsing sphere with 3D text that may change reward,
        create a new goal, or trigger memory recall.
        """
        self._event_counter += 1
        event_id = f"evt_{self._event_counter}_{int(time.time())}"

        text_preview = text[:40] + ("..." if len(text) > 40 else "")
        hue = (hash(text) % 360) / 360.0
        r, g, b = _hue_to_rgb(hue)

        visuals = OrbVisuals(
            base_color=[r, g, b, 0.85],
            glow_color=[min(1, r + 0.3), min(1, g + 0.3), min(1, b + 0.3), 0.5],
            pulse_speed=1.5 + len(text) % 3 * 0.5,
            text_3d=text_preview,
            particle_count=6 + len(text) % 6,
        )

        reward_mod = 0.5 if "?" in text else 0.2
        triggers_recall = len(text) > 20
        creates_goal = any(w in text.lower() for w in ["find", "go", "reach", "get", "collect"])

        event = WorldEvent(
            event_id=event_id,
            event_type="query_orb",
            text=text,
            position=position or [0, 0, 1.0],
            visuals=visuals,
            reward_modifier=reward_mod,
            triggers_memory_recall=triggers_recall,
            creates_new_goal=creates_goal,
        )
        self.active_events[event_id] = event
        return event

    def create_stimulus_event(self, stimulus_type: str,
                              params: dict[str, Any],
                              position: list[float] | None = None) -> WorldEvent:
        self._event_counter += 1
        event_id = f"stim_{self._event_counter}_{int(time.time())}"

        event = WorldEvent(
            event_id=event_id,
            event_type="stimulus",
            text=f"Stimulus: {stimulus_type}",
            position=position or [0, 0, 1.0],
            metadata={"stimulus_type": stimulus_type, "params": params},
        )
        self.active_events[event_id] = event
        return event

    def resolve_event(self, event_id: str, response: str,
                      confidence: float) -> WorldEvent | None:
        event = self.active_events.get(event_id)
        if event is None:
            return None

        event.resolved = True
        event.response = response
        event.confidence = max(0.0, min(1.0, confidence))
        return event

    def get_active_events(self) -> list[WorldEvent]:
        return [e for e in self.active_events.values() if not e.resolved]

    def get_resolved_events(self) -> list[WorldEvent]:
        return [e for e in self.active_events.values() if e.resolved]

    def cleanup_old_events(self, max_age_seconds: float = 300) -> int:
        now = time.time()
        to_remove = [
            eid for eid, e in self.active_events.items()
            if e.resolved and (now - e.created_at) > max_age_seconds
        ]
        for eid in to_remove:
            del self.active_events[eid]
        return len(to_remove)


class ResponseGenerator:
    """
    Generate responses with mandatory confidence and ethical constraints.
    No consciousness claims. Template-based honest responses.
    """

    TEMPLATES = {
        "high_confidence": "Based on available data, {answer}. Current confidence: {confidence:.1%}.",
        "medium_confidence": (
            "Within my data context, {answer}. "
            "Confidence: {confidence:.1%}. Uncertainty remains."
        ),
        "low_confidence": (
            "Insufficient data for a reliable answer regarding '{question}'. "
            "Confidence: {confidence:.1%}. "
            "This assessment is based on limited sensory context."
        ),
        "no_data": (
            "No relevant data available for '{question}'. "
            "Confidence: 0%. Cannot form a reliable assessment."
        ),
    }

    def generate(self, question: str, context_memories: list[dict],
                 agent_state: dict, confidence: float) -> dict[str, Any]:
        """Generate a response with confidence and rationale."""
        if not context_memories:
            template_key = "no_data"
            answer = "no prior experience matches this query"
        elif confidence >= 0.7:
            template_key = "high_confidence"
            answer = self._synthesize_from_memories(context_memories)
        elif confidence >= 0.3:
            template_key = "medium_confidence"
            answer = self._synthesize_from_memories(context_memories)
        else:
            template_key = "low_confidence"
            answer = ""

        response_text = self.TEMPLATES[template_key].format(
            answer=answer,
            question=question,
            confidence=confidence,
        )

        return {
            "text": response_text,
            "confidence": confidence,
            "rationale": f"Based on {len(context_memories)} memory retrievals",
            "memory_count": len(context_memories),
            "template_used": template_key,
        }

    def _synthesize_from_memories(self, memories: list[dict]) -> str:
        if not memories:
            return "no relevant memories found"

        contexts = [m.get("context", "") for m in memories[:3]]
        similarities = [m.get("similarity", 0) for m in memories[:3]]

        parts = []
        for ctx, sim in zip(contexts, similarities):
            if ctx:
                parts.append(f"{ctx} (relevance: {sim:.0%})")

        if parts:
            return "related observations include: " + "; ".join(parts)
        return "some related patterns were detected but details are unclear"


def _hue_to_rgb(h: float) -> tuple[float, float, float]:
    """Convert hue (0-1) to RGB with full saturation and brightness."""
    i = int(h * 6)
    f = h * 6 - i
    q = 1.0 - f
    t = f
    i %= 6
    if i == 0:
        return (1.0, t, 0.0)
    if i == 1:
        return (q, 1.0, 0.0)
    if i == 2:
        return (0.0, 1.0, t)
    if i == 3:
        return (0.0, q, 1.0)
    if i == 4:
        return (t, 0.0, 1.0)
    return (1.0, 0.0, q)


def get_orb_animation_state(event: WorldEvent, current_time: float) -> dict[str, Any]:
    """Get the current animation state of a query orb for rendering."""
    elapsed = current_time - event.created_at
    v = event.visuals

    pulse_phase = math.sin(elapsed * v.pulse_speed * 2 * math.pi)
    scale = 1.0 + pulse_phase * v.pulse_amplitude

    glow_intensity = 0.5 + 0.5 * math.sin(elapsed * v.pulse_speed * math.pi)
    alpha = v.base_color[3] * (0.8 + 0.2 * pulse_phase)

    particles = []
    for i in range(v.particle_count):
        angle = (2 * math.pi * i / v.particle_count) + elapsed * 0.5
        radius = v.glow_radius * (0.5 + 0.5 * math.sin(elapsed * 1.5 + i))
        particles.append({
            "x": math.cos(angle) * radius,
            "y": math.sin(angle) * radius,
            "z": math.sin(elapsed * 2 + i) * 0.3,
            "alpha": 0.3 + 0.3 * math.sin(elapsed * 3 + i),
        })

    return {
        "event_id": event.event_id,
        "position": event.position,
        "scale": scale,
        "alpha": alpha,
        "glow_intensity": glow_intensity,
        "glow_color": v.glow_color,
        "text_3d": v.text_3d,
        "text_scale": v.text_scale,
        "particles": particles,
        "resolved": event.resolved,
    }
