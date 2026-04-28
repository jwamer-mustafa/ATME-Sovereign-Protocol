"""Tests for multilingual pipeline."""

from backend.multilingual.translator import (
    MultilingualPipeline,
    TranslationCache,
    detect_language,
)


def test_detect_arabic():
    """Arabic text is detected correctly."""
    lang, confidence = detect_language("مرحبا كيف حالك")
    assert lang == "ar"
    assert confidence > 0.5


def test_detect_chinese():
    """Chinese text is detected correctly."""
    lang, confidence = detect_language("你好世界")
    assert lang == "zh"
    assert confidence > 0.5


def test_detect_russian():
    """Russian text is detected correctly."""
    lang, confidence = detect_language("Привет мир")
    assert lang == "ru"
    assert confidence > 0.5


def test_detect_english():
    """English text is detected correctly."""
    lang, confidence = detect_language("Hello world how are you")
    assert lang == "en"
    assert confidence > 0.3


def test_detect_empty():
    """Empty text defaults to English."""
    lang, confidence = detect_language("")
    assert lang == "en"
    assert confidence == 0.0


def test_pipeline_process_input_english():
    """English input passes through without translation."""
    pipeline = MultilingualPipeline()
    result = pipeline.process_input("What is the weather like?")
    assert result["english_text"] == "What is the weather like?"
    assert result["source_lang"] == "en"
    assert not result["was_translated"]


def test_pipeline_process_input_arabic():
    """Arabic input is detected and processed."""
    pipeline = MultilingualPipeline()
    result = pipeline.process_input("ما هو الطقس؟")
    assert result["source_lang"] == "ar"
    assert result["lang_name"] == "Arabic"
    assert result["was_translated"]


def test_pipeline_process_output():
    """Output translation back to user language."""
    pipeline = MultilingualPipeline()
    result = pipeline.process_output("The weather is clear.", "ar")
    assert result["target_lang"] == "ar"
    assert result["was_translated"]


def test_pipeline_process_output_english():
    """English output passes through."""
    pipeline = MultilingualPipeline()
    result = pipeline.process_output("Hello", "en")
    assert not result["was_translated"]


def test_translation_cache():
    """Translation cache stores and retrieves results."""
    cache = TranslationCache(max_size=10)
    from backend.multilingual.translator import TranslationResult

    result = TranslationResult(
        original_text="hello",
        translated_text="مرحبا",
        source_lang="en",
        target_lang="ar",
        confidence=0.9,
    )
    cache.put(result)

    cached = cache.get("hello", "en", "ar")
    assert cached is not None
    assert cached.translated_text == "مرحبا"
    assert cached.cached

    # Miss
    miss = cache.get("goodbye", "en", "ar")
    assert miss is None


def test_supported_languages():
    """Supported languages include expected entries."""
    pipeline = MultilingualPipeline()
    langs = pipeline.get_supported_languages()
    assert "en" in langs
    assert "ar" in langs
    assert "zh" in langs
    assert "fr" in langs
