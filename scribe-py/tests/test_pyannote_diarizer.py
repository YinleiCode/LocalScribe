from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scribe_py.diarizers import pyannote_diarizer


@dataclass
class _Turn:
    start: float
    end: float


class _Annotation:
    def __init__(self, rows):
        self.rows = rows

    def itertracks(self, yield_label=False):
        assert yield_label is True
        for start, end, speaker in self.rows:
            yield _Turn(start, end), None, speaker


class _Output:
    def __init__(self, regular, exclusive):
        self.speaker_diarization = regular
        self.exclusive_speaker_diarization = exclusive


def test_community_output_uses_exclusive_assignment_and_regular_overlap(monkeypatch):
    regular = _Annotation([
        (0.0, 4.0, "raw_a"),
        (1.0, 2.0, "raw_b"),
    ])
    exclusive = _Annotation([(0.0, 4.0, "raw_a")])

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, _model, **_kwargs):
            return lambda _audio, **_run_kwargs: _Output(regular, exclusive)

    monkeypatch.setattr(pyannote_diarizer, "_load_pipeline", lambda: _Pipeline)
    result = pyannote_diarizer.diarize(
        Path("audio.wav"),
        [{"start": 0.0, "end": 4.0, "text": "demo"}],
    )

    segment = result.segments[0]
    assert segment.speaker == "SPEAKER_A"
    assert segment.speaker_confidence == 1.0
    assert segment.speaker_overlap_risk is True
    assert segment.speaker_overlap_ratio == 0.25
    assert {item["speaker"] for item in segment.speaker_subsegments or []} == {
        "SPEAKER_A",
        "SPEAKER_B",
    }
    assert result.stats["raw_turns"] == 2
    assert result.stats["exclusive_turns"] == 1


def test_legacy_annotation_output_remains_supported(monkeypatch):
    annotation = _Annotation([
        (0.0, 1.5, "one"),
        (1.5, 3.0, "two"),
    ])

    class _Pipeline:
        @classmethod
        def from_pretrained(cls, _model, **_kwargs):
            return lambda _audio, **_run_kwargs: annotation

    monkeypatch.setattr(pyannote_diarizer, "_load_pipeline", lambda: _Pipeline)
    result = pyannote_diarizer.diarize(
        Path("audio.wav"),
        [{"start": 0.0, "end": 3.0, "text": "demo"}],
    )

    assert result.stats["raw_turns"] == 2
    assert result.stats["exclusive_turns"] == 2
    assert result.segments[0].speaker in {"SPEAKER_A", "SPEAKER_B"}
    assert result.segments[0].speaker_overlap_risk is None
