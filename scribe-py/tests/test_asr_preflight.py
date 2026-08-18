from __future__ import annotations

import sys
from pathlib import Path

_PKG_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(_PKG_SRC))

from scribe_py.ipc import _audio_needs_destructive_enhancement, _select_preflight_recommendation


def _summary(mode: str, *, strong: int, review: int, terms: int = 0, chars: int = 300, risk: str = "high") -> dict:
    return {
        "mode": mode,
        "status": "ok",
        "risk_level": risk,
        "strong_review_count": strong,
        "review_count": review,
        "term_candidate_count": terms,
        "chars": chars,
        "avg_punctuation_ratio": 1.0,
        "score_key": [3, strong, review, terms, 0, 0, 0.05],
    }


def test_preflight_keeps_adaptive_when_enhance_only_has_weak_sample_gain():
    recommended, best, reason = _select_preflight_recommendation(
        [
            _summary("enhance", strong=0, review=8),
            _summary("adaptive", strong=0, review=10),
        ],
        fallback_mode="adaptive",
        audio_quality={
            "risk_level": "high",
            "integrated_lufs": -25.8,
            "risk_reasons": ["多声道会被合并为单声道", "疑似峰值削波/爆音"],
        },
    )

    assert recommended == "adaptive"
    assert best["mode"] == "adaptive"
    assert "保守增强阈值" in reason


def test_preflight_allows_enhance_for_low_loudness_with_decisive_strong_gain():
    recommended, best, reason = _select_preflight_recommendation(
        [
            _summary("enhance", strong=0, review=4),
            _summary("adaptive", strong=3, review=12),
        ],
        fallback_mode="adaptive",
        audio_quality={
            "risk_level": "high",
            "integrated_lufs": -34.0,
            "risk_reasons": ["整体音量过低"],
        },
    )

    assert recommended == "enhance"
    assert best["mode"] == "enhance"
    assert "允许自动使用 enhance" in reason


def test_preflight_treats_noisy_medium_audio_as_needing_enhancement_check():
    assert _audio_needs_destructive_enhancement({
        "risk_level": "medium",
        "integrated_lufs": -22.0,
        "noise_floor_dbfs": -34.0,
        "estimated_snr_db": 12.5,
        "risk_reasons": ["信噪比偏低", "背景噪声偏高"],
    }) is True


def test_preflight_does_not_select_ai_denoise_when_it_fell_back():
    recommended, best, reason = _select_preflight_recommendation(
        [
            {**_summary("ai_denoise", strong=0, review=3), "preprocess_fallback": True},
            _summary("adaptive", strong=2, review=12),
        ],
        fallback_mode="adaptive",
        audio_quality={
            "risk_level": "high",
            "noise_floor_dbfs": -34.0,
            "estimated_snr_db": 12.0,
            "risk_reasons": ["背景噪声明显"],
        },
    )

    assert recommended == "adaptive"
    assert best["mode"] == "adaptive"
    assert "保守增强阈值" in reason


def test_preflight_allows_ai_denoise_for_noisy_audio_with_decisive_gain():
    recommended, best, reason = _select_preflight_recommendation(
        [
            {**_summary("ai_denoise", strong=0, review=3), "preprocess_fallback": False},
            _summary("adaptive", strong=3, review=12),
        ],
        fallback_mode="adaptive",
        audio_quality={
            "risk_level": "high",
            "noise_floor_dbfs": -34.0,
            "estimated_snr_db": 12.0,
            "risk_reasons": ["背景噪声明显"],
        },
    )

    assert recommended == "ai_denoise"
    assert best["mode"] == "ai_denoise"
    assert "允许自动使用 ai_denoise" in reason
