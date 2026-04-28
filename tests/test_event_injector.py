"""Tests for enhanced event injection."""


from backend.simulation.event_injector import (
    EventInjector,
    OrbVisuals,
    get_orb_animation_state,
)


def test_create_question_event_enhanced():
    """Enhanced question events have orb visuals and metadata."""
    injector = EventInjector()
    event = injector.create_question_event("What is this object?")

    assert event.event_type == "query_orb"
    assert event.visuals is not None
    assert isinstance(event.visuals, OrbVisuals)
    assert len(event.visuals.base_color) == 4
    assert event.visuals.pulse_speed > 0
    assert event.visuals.text_3d  # non-empty text preview


def test_question_event_reward_modifier():
    """Questions with '?' get higher reward modifier."""
    injector = EventInjector()
    question = injector.create_question_event("What is this?")
    statement = injector.create_question_event("Tell me about this")

    assert question.reward_modifier > statement.reward_modifier


def test_question_event_triggers_memory():
    """Long questions trigger memory recall."""
    injector = EventInjector()
    short = injector.create_question_event("Hi")
    long = injector.create_question_event("Can you explain what happens when the agent encounters a new environment?")

    assert not short.triggers_memory_recall
    assert long.triggers_memory_recall


def test_question_event_creates_goal():
    """Action-oriented questions create new goals."""
    injector = EventInjector()
    action = injector.create_question_event("Find the red cube")
    info = injector.create_question_event("What is your confidence?")

    assert action.creates_new_goal
    assert not info.creates_new_goal


def test_orb_animation_state():
    """Orb animation state computes pulse and particles."""
    injector = EventInjector()
    event = injector.create_question_event("Test question")
    event.position = [1.0, 2.0, 3.0]

    anim = get_orb_animation_state(event, event.created_at + 0.5)
    assert "scale" in anim
    assert "glow_intensity" in anim
    assert "particles" in anim
    assert len(anim["particles"]) > 0
    assert anim["text_3d"]


def test_orb_visuals_unique_colors():
    """Different questions get different colored orbs."""
    injector = EventInjector()
    e1 = injector.create_question_event("Question one")
    e2 = injector.create_question_event("Completely different topic")

    # Colors should differ (based on text hash)
    assert e1.visuals.base_color != e2.visuals.base_color
