# -*- coding: utf-8 -*-
from __future__ import annotations
import os, pandas as pd
from .base import ITool, ToolResult, get_logger
import numpy as np

log = get_logger("MergeTool")

class MergeTool(ITool):
    def __init__(self, round_name: str, pred_dir: str = "./round_predictions", applied_dir: str = "./round_predictions_applied"):
        self.round_name = round_name
        self.pred_dir = pred_dir
        self.applied_dir = applied_dir

    def run(self) -> ToolResult:
        base = os.path.join(self.pred_dir, f"{self.round_name}_with_predictions.csv")
        applied = os.path.join(self.applied_dir, f"{self.round_name}_position_aware_optimal.csv")
        if not os.path.exists(base):    return ToolResult(False, f"base 없음: {base}")
        if not os.path.exists(applied): return ToolResult(False, f"applied 없음: {applied}")

        dfb = pd.read_csv(base, low_memory=False)
        dfa = pd.read_csv(applied, low_memory=False)

        # 1) 병합 키 결정
        merge_key = None
        if "alert_id" in dfb.columns and "alert_id" in dfa.columns:
            merge_key = "alert_id"
        elif "__row_id" in dfb.columns and "__row_id" in dfa.columns:
            merge_key = "__row_id"
        else:
            # __row_id 자동 생성 시도
            if "__row_id" not in dfb.columns:
                dfb["__row_id"] = np.arange(len(dfb), dtype=np.int64)
            if "__row_id" not in dfa.columns:
                if len(dfa) == len(dfb):
                    dfa["__row_id"] = np.arange(len(dfa), dtype=np.int64)
                else:
                    # 복합키 마지막 안전망
                    candidate_keys = ["capsule_duration","A_ip","B_ip","A_port","B_port"]
                    if all(k in dfb.columns for k in candidate_keys) and all(k in dfa.columns for k in candidate_keys):
                        merge_key = candidate_keys
                    else:
                        return ToolResult(False, "병합 키(alert_id/__row_id/복합키) 없음")
            if merge_key is None:
                merge_key = "__row_id"

        # 2) applied에서 유지할 컬럼
        keep = [
            "adjusted_label","feedback_applied",
            "applied_from_case","applied_from_file","applied_reason","applied_confidence",
            "applied_similarity_score","applied_similarity_ip","applied_similarity_cosine",
            "applied_similarity_overlap","applied_common_features",
        ]

        # ⬇⬇⬇ 여기 중요: 병합키를 우측 keep 컬럼에 포함시켜야 함
        if isinstance(merge_key, list):
            merge_keys = merge_key
            keep = merge_keys + keep        # <-- 추가
        else:
            merge_keys = [merge_key]
            keep = [merge_key] + keep

        for c in keep:
            if c not in dfa.columns:
                dfa[c] = "" if c not in ("applied_similarity_score","applied_similarity_ip",
                                         "applied_similarity_cosine","applied_similarity_overlap") else 0.0

        merged = dfb.merge(dfa[keep], on=merge_keys, how="left")

        if "adjusted_label" not in merged.columns:
            merged["adjusted_label"] = merged.get("predicted_label","")
        merged["final_label"] = merged["adjusted_label"].where(
            merged["adjusted_label"].notna() & (merged["adjusted_label"]!=""),
            merged.get("predicted_label","")
        )

        out = os.path.join(self.pred_dir, f"{self.round_name}_with_predictions_applied.csv")
        merged.to_csv(out, index=False, encoding="utf-8")
        log.info(f"병합 저장: {out}")
        return ToolResult(True, "ok", output_path=out)
