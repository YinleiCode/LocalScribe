"""OpenAI-compatible LLM corrector. Works with DeepSeek / OpenAI / Claude API / Ollama / vLLM.

Implements the **B-style two-pass** strategy:
  Pass 1: single LLM call to scan full text → glossary of proper nouns
  Pass 2: parallel batched correction with the glossary injected into each system prompt

Pass 2 is parallelised with a `ThreadPoolExecutor` (default 5 workers) which gives
roughly 5x throughput while keeping per-batch quality identical (each batch is an
independent stateless LLM call).

Supports cooperative pause/cancel via a `CorrectionControl` object: the corrector
checks `is_paused()` / `is_cancelled()` between batches and waits / aborts.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from openai import OpenAI

from ..control import CorrectionControl
from ..core.types import Segment
from . import prompts
from .base import Corrector, ProgressCallback


class OpenAICompatibleCorrector(Corrector):
    name = "openai_compatible"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        mode: str = "medium",
        batch_size: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 8192,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        use_glossary: bool = True,
        glossary_text_cap: int = 200_000,
        concurrency: int = 15,
        language: str | None = None,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url
        self.model = model
        self.mode = mode
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.use_glossary = use_glossary
        self.glossary_text_cap = glossary_text_cap
        self.concurrency = max(1, concurrency)
        self.language = language
        self.base_system_prompt = prompts.get(mode, language)
        # Filled in by `correct()` once Pass 1 (glossary) completes.
        self._effective_system_prompt = self.base_system_prompt
        self.last_glossary: list[dict] = []
        self.last_cancelled: bool = False

    # ---- Pass 1: glossary extraction ----

    def _extract_glossary(self, segments: list[Segment], context_hint: str = "") -> list[dict]:
        """Single LLM call. Returns list of glossary entries, or [] on failure."""
        # Concatenate plain text only (no timestamps) for compactness.
        full_text = "".join(s.text for s in segments)
        if len(full_text) > self.glossary_text_cap:
            full_text = full_text[: self.glossary_text_cap]
        payload = {"context_hint": context_hint, "text": full_text}
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompts.get_glossary_prompt(self.language)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            data = json.loads(rsp.choices[0].message.content)
            glossary = data.get("glossary", [])
            if not isinstance(glossary, list):
                return []
            # Sort by freq desc, cap to 80
            glossary = [
                g for g in glossary
                if isinstance(g, dict) and isinstance(g.get("term"), str) and g["term"].strip()
            ]
            glossary.sort(key=lambda g: g.get("freq", 0), reverse=True)
            return glossary[:80]
        except Exception:
            return []

    # ---- Pass 2: batched correction (uses _effective_system_prompt) ----

    def _correct_batch(self, batch: list[Segment], context_hint: str) -> list[Segment]:
        payload = {
            "context_hint": context_hint,
            "segments": [{"idx": i, "text": s.text} for i, s in enumerate(batch)],
        }
        rsp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._effective_system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        data = json.loads(rsp.choices[0].message.content)
        by_idx = {s["idx"]: s["text"] for s in data.get("segments", [])}
        out: list[Segment] = []
        for i, src in enumerate(batch):
            corrected = by_idx.get(i, src.text).strip() or src.text
            out.append(Segment(
                start=src.start, end=src.end, text=corrected,
                original_text=src.text, speaker=src.speaker,
            ))
        return out

    # ---- Orchestration ----

    def correct(
        self,
        segments: list[Segment],
        context_hint: str = "",
        on_progress: ProgressCallback | None = None,
        control: CorrectionControl | None = None,
    ) -> list[Segment]:
        """Run two-pass correction with parallel Pass 2.

        If `control` is provided, the run respects `pause` / `cancel` flags:
        - `pause`: blocks new batch dispatches; in-flight batches finish naturally.
        - `cancel`: stops dispatching new batches; collects whatever is done; remaining
          segments are returned with original text. Sets `self.last_cancelled = True`.
        """
        total = len(segments)
        self.last_cancelled = False
        self.last_glossary = []

        if total == 0:
            if on_progress:
                on_progress({"current": 0, "total": 0, "stage": "done"})
            return []

        # Pass 1: glossary (single call — not parallelised)
        glossary: list[dict] = []
        if self.use_glossary:
            if on_progress:
                on_progress({"current": 0, "total": total, "stage": "glossary"})
            if control:
                control.wait_if_paused()
                if control.is_cancelled():
                    self.last_cancelled = True
                    return [Segment(s.start, s.end, s.text, s.text, speaker=s.speaker) for s in segments]
            glossary = self._extract_glossary(segments, context_hint)
            self.last_glossary = glossary
            if on_progress:
                on_progress({
                    "current": 0,
                    "total": total,
                    "stage": "glossary_done",
                    "glossary_count": len(glossary),
                })

        self._effective_system_prompt = prompts.with_glossary(self.base_system_prompt, glossary, self.language)

        # Pass 2: parallel batched correction
        batches: list[tuple[int, list[Segment]]] = []
        for i in range(0, total, self.batch_size):
            batches.append((i, segments[i : i + self.batch_size]))

        results: dict[int, list[Segment]] = {}

        def _process_batch_safe(idx: int, batch: list[Segment]) -> tuple[int, list[Segment]]:
            try:
                return idx, self._correct_batch(batch, context_hint)
            except Exception:  # noqa: BLE001 — keep going with originals on any failure
                return idx, [
                    Segment(s.start, s.end, s.text, original_text=s.text, speaker=s.speaker)
                    for s in batch
                ]

        # We submit work in a controlled fashion so that pause stops *new* dispatches
        # while in-flight ones complete naturally.
        executor = ThreadPoolExecutor(max_workers=self.concurrency, thread_name_prefix="correct")
        in_flight: dict = {}  # Future -> (idx, batch)
        next_to_submit = 0
        completed_count = 0

        try:
            while next_to_submit < len(batches) or in_flight:
                # Honour cancellation
                if control and control.is_cancelled():
                    self.last_cancelled = True
                    break

                # Honour pause: don't submit new work; if nothing in flight, block.
                if control and control.is_paused() and not in_flight:
                    while control.is_paused() and not control.is_cancelled():
                        time.sleep(0.3)
                    continue

                # Top up the executor up to concurrency (unless paused/cancelled)
                while (
                    next_to_submit < len(batches)
                    and len(in_flight) < self.concurrency
                    and not (control and (control.is_paused() or control.is_cancelled()))
                ):
                    idx, batch = batches[next_to_submit]
                    fut = executor.submit(_process_batch_safe, idx, batch)
                    in_flight[fut] = (idx, batch)
                    next_to_submit += 1

                # Wait briefly for any in-flight to complete
                if in_flight:
                    done, _pending = wait(
                        list(in_flight.keys()),
                        timeout=0.5,
                        return_when=FIRST_COMPLETED,
                    )
                    for fut in done:
                        _, _batch = in_flight.pop(fut)
                        idx, batch_out = fut.result()
                        results[idx] = batch_out
                        completed_count += 1
                        if on_progress:
                            on_progress({
                                "current": min(total, completed_count * self.batch_size),
                                "total": total,
                                "stage": "correcting",
                                "batches_done": completed_count,
                                "batches_total": len(batches),
                                "concurrency": self.concurrency,
                            })
        finally:
            # If cancelling, tell still-pending futures to abort if possible.
            for fut in list(in_flight.keys()):
                fut.cancel()
            executor.shutdown(wait=True)

        # Reassemble in original idx order. Any missing batch (cancelled before
        # dispatch) is filled with original text so output length matches input.
        out: list[Segment] = []
        for idx, batch in batches:
            batch_out = results.get(idx)
            if batch_out is None:
                batch_out = [
                    Segment(s.start, s.end, s.text, original_text=s.text, speaker=s.speaker)
                    for s in batch
                ]
            out.extend(batch_out)

        if on_progress:
            stage = "cancelled" if self.last_cancelled else "done"
            on_progress({"current": total, "total": total, "stage": stage})
        return out
