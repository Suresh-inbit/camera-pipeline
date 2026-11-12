"""isp - a small, readable camera-pipeline and post-processing toolkit (EE 5176).

Conventions (full text in docs/conventions.md)
----------------------------------------------
* Images are ``float32``.  RGB images are ``(H, W, 3)``; a RAW mosaic is ``(H, W)``.
* Values are *linear* (proportional to light) and scene-referred in ``[0, 1]`` up to
  the tone-mapping stage, and display-referred (sRGB-encoded) after it.
* Every image travels with a ``meta`` dict; ``meta['domain']`` says what the numbers
  mean: ``scene | electrons | dn | linear | display | bytes``.
* Library functions are pure and never plot.
"""
import importlib

from . import utils, scenes, pipeline
from .pipeline import Pipeline, Stage, StageRecord, PipelineResult

_METHOD_MODULES = ("sensor", "cfa", "demosaic", "whitebalance", "tonemap",
                   "denoise", "sharpen", "jpeg", "display", "io")

__all__ = ["utils", "scenes", "pipeline", "Pipeline", "Stage", "StageRecord",
           "PipelineResult", *_METHOD_MODULES]


def __getattr__(name):
    """Import method modules on first use (``isp.sensor`` etc.)."""
    if name in _METHOD_MODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module 'isp' has no attribute {name!r}")
