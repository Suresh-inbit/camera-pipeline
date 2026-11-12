"""The camera pipeline as an inspectable list of stages.

A *stage* is any function ``fn(image, meta, **params) -> (image, meta)``.
``Pipeline.run`` applies the enabled stages in order and keeps a ``StageRecord`` for
every stage, so each intermediate image can be displayed, measured and compared.
Stages can be switched off, re-parameterised and re-ordered ("the processing order
may vary", lecture 2 slide 3).

The default camera (scene -> sensor -> ADC -> processing -> storage) is built by
``default_pipeline`` once the method modules exist.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

GROUPS = ("scene", "sensor", "adc", "processing", "storage")   # the boxes of slide 3


@dataclass
class Stage:
    """One step of the pipeline.

    name    unique id, e.g. ``'demosaic'``
    fn      ``fn(image, meta, **params) -> (image, meta)``; must not modify its inputs
    params  keyword arguments for ``fn``
    enabled disabled stages pass their input through unchanged
    group   one of ``GROUPS`` (which box of the slide-3 diagram the stage belongs to)
    label   human-readable name for diagrams (defaults to ``name``)
    """
    name: str
    fn: Callable
    params: dict = field(default_factory=dict)
    enabled: bool = True
    group: str = "processing"
    label: str | None = None

    def __post_init__(self):
        if self.group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {self.group!r}")
        if self.label is None:
            self.label = self.name


@dataclass
class StageRecord:
    """What a stage produced: its output, the meta data and how long it took."""
    name: str
    label: str
    group: str
    image: np.ndarray
    meta: dict
    params: dict
    enabled: bool
    seconds: float
    metrics: dict = field(default_factory=dict)

    @property
    def summary(self) -> dict:
        a = self.image
        return dict(name=self.name, group=self.group, enabled=self.enabled,
                    domain=self.meta.get("domain"), shape=tuple(a.shape), dtype=str(a.dtype),
                    min=float(np.min(a)), max=float(np.max(a)), seconds=self.seconds, **self.metrics)


class PipelineResult:
    """Ordered stage records; index by position or by stage name."""

    def __init__(self, records: list):
        self.records = list(records)

    def __getitem__(self, key) -> StageRecord:
        if isinstance(key, str):
            for r in self.records:
                if r.name == key:
                    return r
            raise KeyError(key)
        return self.records[key]

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    @property
    def names(self) -> list:
        return [r.name for r in self.records]

    @property
    def final(self) -> StageRecord:
        return self.records[-1]

    def table(self) -> list:
        """One summary dict per stage (handy for ``pandas.DataFrame``)."""
        return [r.summary for r in self.records]


class Pipeline:
    """An ordered, editable list of stages."""

    def __init__(self, stages: list):
        self.stages = list(stages)
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")

    # ---- editing ---------------------------------------------------------
    @property
    def names(self) -> list:
        return [s.name for s in self.stages]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def __getitem__(self, name: str) -> Stage:
        return self.stages[self.index(name)]

    def copy(self) -> "Pipeline":
        """Independent copy (stage params are deep-copied, functions are shared)."""
        return Pipeline([Stage(s.name, s.fn, copy.deepcopy(s.params), s.enabled, s.group, s.label)
                         for s in self.stages])

    def set(self, name: str, **params) -> "Pipeline":
        """Update parameters of a stage.  Returns ``self`` for chaining."""
        self[name].params.update(params)
        return self

    def enable(self, name: str, enabled: bool = True) -> "Pipeline":
        self[name].enabled = bool(enabled)
        return self

    def move(self, name: str, before: str | None = None, after: str | None = None) -> "Pipeline":
        """Move a stage directly before or after another one."""
        if (before is None) == (after is None):
            raise ValueError("give exactly one of before= / after=")
        stage = self.stages.pop(self.index(name))
        i = self.index(before) if before is not None else self.index(after) + 1
        self.stages.insert(i, stage)
        return self

    # ---- running ---------------------------------------------------------
    def run(self, image, meta: dict | None = None, metrics: Callable | None = None) -> PipelineResult:
        """Run all stages on ``image``.

        meta     initial meta dict (default ``{'domain': 'scene'}``)
        metrics  optional ``metrics(image, meta, stage_name) -> dict`` evaluated on every
                 stage output (e.g. PSNR against the ground-truth scene)
        """
        img = np.asarray(image)
        meta = dict(meta) if meta is not None else {"domain": "scene"}
        records = []
        for s in self.stages:
            t0 = time.perf_counter()
            if s.enabled:
                img, meta = s.fn(img, dict(meta), **s.params)
                meta = dict(meta)
            dt = time.perf_counter() - t0
            m = dict(metrics(img, meta, s.name)) if metrics is not None else {}
            records.append(StageRecord(s.name, s.label, s.group, img, dict(meta),
                                       copy.deepcopy(s.params), s.enabled, dt, m))
        return PipelineResult(records)
