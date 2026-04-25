"""Tests for the ecology layer."""

from backend.simulation.ecology import (
    CurriculumState,
    DayNightCycle,
    EcologyManager,
    TimeOfDay,
)


def test_day_night_cycle_update():
    """Day/night cycle progresses correctly."""
    cycle = DayNightCycle(time_scale=60.0, current_time=0.25)
    assert cycle.time_of_day == TimeOfDay.DAWN

    # Advance to midday
    cycle.update(dt=18.0)  # 0.25 + 18/60 = 0.55
    assert cycle.time_of_day == TimeOfDay.DAY
    assert cycle.sun_intensity > 0.5

    # Advance to night
    cycle.current_time = 0.9
    cycle.update(dt=0.0)
    assert cycle.time_of_day == TimeOfDay.NIGHT
    assert cycle.sun_intensity < 0.5


def test_day_night_sky_color():
    """Sky color changes with time of day."""
    cycle = DayNightCycle()
    cycle.current_time = 0.5  # midday
    cycle.update(dt=0.0)
    day_sky = cycle.sky_color.copy()

    cycle.current_time = 0.95  # night
    cycle.update(dt=0.0)
    night_sky = cycle.sky_color.copy()

    assert day_sky != night_sky


def test_ecology_spawn_resource():
    """Resources can be spawned and collected."""
    eco = EcologyManager(seed=42)
    res = eco.spawn_resource(position=[1.0, 2.0, 0.3], resource_type="energy", value=2.0)
    assert res.resource_id == "res_1"
    assert res.resource_type == "energy"
    assert not res.collected


def test_ecology_spawn_npc():
    """NPCs can be spawned with different behaviors."""
    eco = EcologyManager(seed=42)
    npc = eco.spawn_npc(position=[0, 0, 0.5], behavior="wander")
    assert npc.npc_id == "npc_1"
    assert npc.behavior == "wander"


def test_ecology_resource_collection():
    """Resources are collected when agent is close enough."""
    eco = EcologyManager(seed=42)
    eco.spawn_resource(position=[1.0, 0.0, 0.3])

    # Agent too far
    result = eco.update(dt=0.1, agent_position=[5.0, 5.0, 0.0], current_time=0.0)
    assert len(result["events"]) == 0

    # Agent close enough
    result = eco.update(dt=0.1, agent_position=[1.0, 0.0, 0.0], current_time=1.0)
    collected_events = [e for e in result["events"] if e["type"] == "resource_collected"]
    assert len(collected_events) == 1


def test_curriculum_difficulty_adjustment():
    """Difficulty adjusts based on success rate."""
    curriculum = CurriculumState(difficulty=0.5, window_size=10)

    # All successes → difficulty should increase
    for _ in range(15):
        curriculum.record_outcome(True)
    assert curriculum.difficulty > 0.5

    # Reset and all failures → difficulty should decrease
    curriculum.difficulty = 0.5
    curriculum.success_window = []
    for _ in range(15):
        curriculum.record_outcome(False)
    assert curriculum.difficulty < 0.5


def test_ecology_get_state():
    """State serialization works correctly."""
    eco = EcologyManager(seed=42)
    eco.spawn_resource(position=[1, 0, 0])
    eco.spawn_npc(position=[2, 0, 0], behavior="patrol")
    state = eco.get_state()
    assert len(state["resources"]) == 1
    assert len(state["npcs"]) == 1
    assert "day_night" in state
    assert "weather" in state


def test_ecology_reward_modifier():
    """Reward modifier changes with conditions."""
    eco = EcologyManager(seed=42)
    base_mod = eco.get_reward_modifier()
    assert base_mod >= 1.0

    # Night should increase modifier
    eco.day_night.current_time = 0.9
    eco.day_night.update(dt=0.0)
    night_mod = eco.get_reward_modifier()
    assert night_mod >= base_mod
