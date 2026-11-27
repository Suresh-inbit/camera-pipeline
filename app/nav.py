"""The clickable pipeline diagram: a zero-build bidirectional Streamlit component.

``app/pipeline_nav/index.html`` implements the component protocol by hand (see the comment at its
top for the message types, read from the installed Streamlit).  ``pipeline_nav(payload)`` returns
``None`` before the first click (and always under ``AppTest``), then ``{'stage': id, 'nonce': str}``
for the last click; ``handle_click`` acts only when the nonce is new.
"""
from __future__ import annotations

import pathlib

import streamlit as st
import streamlit.components.v1 as components

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent / "pipeline_nav"
_component = components.declare_component("pipeline_nav", path=str(COMPONENT_DIR))


def pipeline_nav(payload: dict, key: str = "pipeline_nav"):
    """Render the diagram; returns the last click value (dict) or None."""
    return _component(nodes=payload["nodes"], groups=payload["groups"], current=payload["current"],
                      key=key, default=None)


def new_click(value, seen):
    """The stage id of a click that has not been handled yet, else None (pure function)."""
    if not isinstance(value, dict):
        return None
    nonce, stage = value.get("nonce"), value.get("stage")
    if not nonce or nonce == seen or not isinstance(stage, str):
        return None
    return stage
