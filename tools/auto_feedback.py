# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
from typing import Optional
from .base import ITool, ToolResult, get_logger

log = get_logger("AutoFeedbackTool")
KST = timezone(timedelta(hours=9))

class AutoFeedbackTool(ITool):
    """
    입력: ./feedback_cases/{Round}_low_confidence_cases.csv
    동작: 미검토 False 중 확신도 낮은 순 Top-N 또는 하위 % → 실제 라벨 복사
    출력: 입력 파일 업데이트(덮어쓰기)
    """
    def __init__(self,
                 round_name: str,
                 feedback_dir: str = "./feedback_cases",
                 top_n: Optional[int] = 300,
                 percent: Optional[float] = None,
                 use_kst: bool = True):
        self.round_name = round_name
        self.feedback_dir = feedback_dir
        self.top_n = top_n
        self.percent = percent
        self.use_kst = use_kst

    def run(self) -> ToolResult:
        path = os.path.join(self.feedback_dir, f"{self.round_name}_low_confidence_cases.csv")
        if not os.path.exists(path):
            return ToolResult(False, f"피드백 케이스 파일 없음: {path}")

        df = pd.read_csv(path, low_memory=False)
        score_col = "confidence_score" if "confidence_score" in df.columns else (
            "prediction_confidence" if "prediction_confidence" in df.columns else None
        )
        if not score_col:
            return ToolResult(False, "confidence_score/prediction_confidence 없음")

        work = df[df.get("reviewed", False) == False].copy()
        if work.empty:
            log.info("미검토 없음 → 스킵")
            return ToolResult(True, "no-op", output_path=path)

        sort_cols = [score_col]
        if "attack_probability" in work.columns: sort_cols.append("attack_probability")
        if "case_id" in work.columns: sort_cols.append("case_id")
        work = work.sort_values(sort_cols, ascending=[True]*len(sort_cols), kind="mergesort")

        if self.percent is not None and self.top_n is not None:
            return ToolResult(False, "percent와 top_n 동시 지정 불가")
        if self.percent is not None:
            if not (0 < self.percent <= 100): return ToolResult(False, "percent는 0~100")
            k = max(1, int(len(work) * (self.percent/100.0)))
        else:
            k = min(int(self.top_n or 50), len(work))

        picks = work.head(k).copy()
        if picks.empty:
            return ToolResult(True, "선택 0", output_path=path)

        if "label" not in df.columns:
            return ToolResult(False, "label 컬럼 없음 → 실제 라벨 복사 불가")

        case_ids = set(picks["case_id"])
        idx = df["case_id"].isin(case_ids)
        today = datetime.now(KST if self.use_kst else None).strftime("%Y-%m-%d")

        for c, default in [
            ("feedback_label",""),("feedback_confidence",""),
            ("feedback_reason",""),("needs_review",True),
            ("reviewed",False),("review_date",""),
        ]:
            if c not in df.columns: df[c] = default

        df.loc[idx, "feedback_label"] = df.loc[idx, "label"]
        df.loc[idx, "feedback_confidence"] = 5
        df.loc[idx, "feedback_reason"] = "(자동) 실제 라벨 기반 자동 피드백"
        df.loc[idx, "reviewed"] = True
        df.loc[idx, "needs_review"] = False
        df.loc[idx, "review_date"] = today

        df.to_csv(path, index=False, encoding="utf-8")
        log.info(f"자동 플래그 적용 {int(idx.sum())}/{k} → {path}")
        return ToolResult(True, "ok", output_path=path, data={"updated": int(idx.sum()), "selected": k})
