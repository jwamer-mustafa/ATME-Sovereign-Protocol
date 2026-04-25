"""
Event Injection: Transform user questions/interactions into world entities.
Questions become physical objects (query_orbs) that the agent can approach.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


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


class EventInjector:
    """Manages the lifecycle of injected events in the simulation."""

    def __init__(self):
        self.active_events: dict[str, WorldEvent] = {}
        self._event_counter = 0

    def create_question_event(self, text: str,
                              position: list[float] | None = None) -> WorldEvent:
        """
        Create a question event that will be injected as a query_orb.
        The simulation will create the physical entity.
        """
        self._event_counter += 1
        event_id = f"evt_{self._event_counter}_{int(time.time())}"

        event = WorldEvent(
            event_id=event_id,
            event_type="query_orb",
            text=text,
            position=position or [0, 0, 1.0],
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
