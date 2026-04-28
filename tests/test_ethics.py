"""Tests for ethical safeguards."""


from backend.ethics.safeguards import (
    ensure_confidence_display,
    format_ethical_response,
    sanitize_response,
)


def test_sanitize_forbidden_patterns():
    result = sanitize_response("I feel happy about this")
    assert not result.is_safe
    assert "Within my data processing" in result.sanitized_text
    assert len(result.violations) > 0


def test_sanitize_safe_text():
    result = sanitize_response("Based on available data, the answer is 42")
    assert result.is_safe
    assert len(result.violations) == 0


def test_multiple_violations():
    result = sanitize_response("I feel and I want and I believe this is true")
    assert not result.is_safe
    assert len(result.violations) == 3


def test_ensure_confidence():
    text = ensure_confidence_display("Answer is 42", 0.75)
    assert "Confidence: 75.0%" in text


def test_ensure_confidence_already_present():
    text = ensure_confidence_display("Answer is 42. Confidence: 80%", 0.75)
    assert text.count("Confidence") == 1


def test_format_ethical_response():
    resp = format_ethical_response(
        "I think the answer is 42",
        confidence=0.5,
        uncertainty_factors=["limited data"],
    )
    assert resp["confidence"] == 0.5
    assert resp["uncertainty"] == 0.5
    assert "According to my analysis" in resp["text"]
    assert resp["ethical_check"]["violations_found"] > 0
    assert "disclaimer" in resp
