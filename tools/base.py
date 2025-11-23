# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any, Dict, Optional
import os, sys, json, time, logging

# ── 공용 로거 ──────────────────────────────────────────────
def get_logger(name: str = "pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger

# ── Tool 인터페이스 ───────────────────────────────────────
class ITool(Protocol):
    def run(self, **kwargs) -> "ToolResult": ...

@dataclass
class ToolResult:
    ok: bool
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    output_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

# ── 유틸 ───────────────────────────────────────────────────
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p
