"""Compatibility shim for running `uvicorn app.main:app` from repo root.

Canonical backend package lives in `backend/app`. Running from `backend/` still
imports that package directly; running from repo root imports this shim, which
extends the package search path to include `backend/app`.
"""
from pathlib import Path

_backend_app = Path(__file__).resolve().parent.parent / 'backend' / 'app'
if _backend_app.exists():
    __path__.append(str(_backend_app))
