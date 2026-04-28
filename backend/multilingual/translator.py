"""
Multilingual support: language detection + translation pipeline.
Supports input/output in any language via detection → translate → process → translate back.

Uses lightweight approaches:
- Language detection via character analysis + simple heuristics (no heavy deps)
- Translation stubs with caching (real NLLB/M2M100 models can be plugged in)
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    cached: bool = False


# Unicode script ranges for language detection
_SCRIPT_RANGES: dict[str, list[tuple[int, int]]] = {
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "zh": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x2F00, 0x2FDF)],
    "ja": [(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF)],
    "ko": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "hi": [(0x0900, 0x097F)],
    "th": [(0x0E00, 0x0E7F)],
    "ru": [(0x0400, 0x04FF)],
    "he": [(0x0590, 0x05FF)],
    "fa": [(0x0600, 0x06FF), (0xFB50, 0xFDFF)],
}

_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "th": "Thai",
    "ru": "Russian",
    "he": "Hebrew",
    "fa": "Persian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "tr": "Turkish",
}


def detect_language(text: str) -> tuple[str, float]:
    """
    Detect the language of input text using Unicode script analysis.
    Returns (language_code, confidence).
    """
    if not text or not text.strip():
        return ("en", 0.0)

    text_clean = re.sub(r'[0-9\s\W]+', '', text)
    if not text_clean:
        return ("en", 0.5)

    script_counts: dict[str, int] = {}
    latin_count = 0
    total_chars = 0

    for char in text_clean:
        code_point = ord(char)
        total_chars += 1
        found_script = False

        for lang, ranges in _SCRIPT_RANGES.items():
            for start, end in ranges:
                if start <= code_point <= end:
                    script_counts[lang] = script_counts.get(lang, 0) + 1
                    found_script = True
                    break
            if found_script:
                break

        if not found_script:
            cat = unicodedata.category(char)
            if cat.startswith('L'):
                latin_count += 1

    if total_chars == 0:
        return ("en", 0.5)

    if script_counts:
        best_lang = max(script_counts, key=script_counts.get)  # type: ignore[arg-type]
        confidence = script_counts[best_lang] / total_chars
        return (best_lang, min(0.99, confidence))

    if latin_count / total_chars > 0.5:
        return _detect_latin_language(text_clean)

    return ("en", 0.5)


def _detect_latin_language(text: str) -> tuple[str, float]:
    """Simple heuristic for detecting Latin-script languages."""
    text_lower = text.lower()

    lang_patterns: dict[str, list[str]] = {
        "fr": ["les", "des", "une", "est", "que", "dans", "pour", "avec", "cette"],
        "de": ["der", "die", "das", "und", "ist", "ein", "nicht", "mit", "auf"],
        "es": ["los", "las", "una", "que", "del", "por", "con", "para", "esta"],
        "pt": ["uma", "dos", "das", "que", "com", "para", "por", "como", "mais"],
        "it": ["della", "che", "una", "per", "con", "sono", "come", "questo"],
        "tr": ["bir", "olan", "ile", "için", "olan", "gibi", "daha", "sonra"],
    }

    best_lang = "en"
    best_score = 0
    words = set(re.findall(r'\b\w+\b', text_lower))

    for lang, patterns in lang_patterns.items():
        score = sum(1 for p in patterns if p in words)
        if score > best_score:
            best_score = score
            best_lang = lang

    confidence = min(0.9, best_score * 0.15 + 0.3) if best_score > 0 else 0.6
    return (best_lang, confidence)


class TranslationCache:
    """LRU cache for translations."""

    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict[str, TranslationResult] = OrderedDict()
        self._max_size = max_size

    def _key(self, text: str, source: str, target: str) -> str:
        raw = f"{source}:{target}:{text}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, text: str, source: str, target: str) -> TranslationResult | None:
        key = self._key(text, source, target)
        if key in self._cache:
            self._cache.move_to_end(key)
            result = self._cache[key]
            result.cached = True
            return result
        return None

    def put(self, result: TranslationResult) -> None:
        key = self._key(result.original_text, result.source_lang, result.target_lang)
        self._cache[key] = result
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


class MultilingualPipeline:
    """
    Full multilingual pipeline:
    Input (any lang) → Detect → Translate to English → Process → Translate back → Output
    """

    def __init__(self):
        self._cache = TranslationCache()

    def process_input(self, text: str) -> dict[str, Any]:
        """
        Process user input in any language.
        Returns the English translation with metadata.
        """
        lang, confidence = detect_language(text)

        if lang == "en":
            return {
                "original_text": text,
                "english_text": text,
                "source_lang": "en",
                "lang_name": "English",
                "detection_confidence": confidence,
                "was_translated": False,
            }

        cached = self._cache.get(text, lang, "en")
        if cached:
            return {
                "original_text": text,
                "english_text": cached.translated_text,
                "source_lang": lang,
                "lang_name": _LANG_NAMES.get(lang, lang),
                "detection_confidence": confidence,
                "was_translated": True,
                "cached": True,
            }

        translated = self._translate(text, lang, "en")

        self._cache.put(translated)

        return {
            "original_text": text,
            "english_text": translated.translated_text,
            "source_lang": lang,
            "lang_name": _LANG_NAMES.get(lang, lang),
            "detection_confidence": confidence,
            "was_translated": True,
            "cached": False,
        }

    def process_output(self, text: str, target_lang: str) -> dict[str, Any]:
        """
        Translate English response back to the user's language.
        """
        if target_lang == "en":
            return {
                "original_text": text,
                "translated_text": text,
                "target_lang": "en",
                "lang_name": "English",
                "was_translated": False,
            }

        cached = self._cache.get(text, "en", target_lang)
        if cached:
            return {
                "original_text": text,
                "translated_text": cached.translated_text,
                "target_lang": target_lang,
                "lang_name": _LANG_NAMES.get(target_lang, target_lang),
                "was_translated": True,
                "cached": True,
            }

        translated = self._translate(text, "en", target_lang)
        self._cache.put(translated)

        return {
            "original_text": text,
            "translated_text": translated.translated_text,
            "target_lang": target_lang,
            "lang_name": _LANG_NAMES.get(target_lang, target_lang),
            "was_translated": True,
            "cached": False,
        }

    def _translate(self, text: str, source: str, target: str) -> TranslationResult:
        """
        Translate text between languages.
        This is a stub that preserves the text — plug in NLLB-200 or M2M100 for real translation.
        In production, replace this with:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        """
        return TranslationResult(
            original_text=text,
            translated_text=text,
            source_lang=source,
            target_lang=target,
            confidence=0.8,
        )

    @staticmethod
    def get_supported_languages() -> dict[str, str]:
        """Get map of supported language codes to names."""
        return dict(_LANG_NAMES)
