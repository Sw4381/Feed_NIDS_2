# tools/detection.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from .base import ITool, ToolResult, ensure_dir, get_logger

log = get_logger("DetectionTool")


# ---------------------------
# 유틸: 입력 라운드 탐색
# ---------------------------
def _list_round_inputs(rounds_directory: str) -> List[Tuple[str, str]]:
    """
    rounds_directory 안에서 Round_* 입력 파일을 찾아 (round_name, path) 리스트 반환.
    우선순위: {Round}.csv > {Round}_raw.csv > 기타 {Round}*.csv
    """
    cand = sorted(glob.glob(os.path.join(rounds_directory, "Round_**.csv")))
    if not cand:
        return []

    # 라운드명 추출: 파일명에서 접미어 삭제
    pairs: dict[str, List[str]] = {}
    for p in cand:
        bn = os.path.basename(p)
        # Round_XYZ_... 형태에서 Round_XYZ만 남김
        rn = bn.split(".csv")[0]
        # 가장 일반적인 형태 우선
        if rn.endswith("_raw"):
            base = rn[:-4]
        else:
            base = rn
        # base가 Round_* 로 시작하는 것만
        if not base.startswith("Round_"):
            continue
        pairs.setdefault(base, []).append(p)

    out: List[Tuple[str, str]] = []
    for base, paths in pairs.items():
        # 우선순위: {Round}.csv > {Round}_raw.csv > 기타
        pref = [
            os.path.join(rounds_directory, f"{base}.csv"),
            os.path.join(rounds_directory, f"{base}_raw.csv"),
        ]
        chosen = None
        for pr in pref:
            if pr in paths and os.path.exists(pr):
                chosen = pr
                break
        if chosen is None:
            # 기타 후보 중 첫 번째 사용
            chosen = sorted(paths)[0]
        out.append((base, chosen))
    return sorted(out, key=lambda x: x[0])


# ---------------------------
# 유틸: 모델 로드 & feature 결정
# ---------------------------
def _load_model_and_features(model_path: str) -> Tuple[object, List[str]]:
    """
    joblib으로 저장된 모델 또는 dict(model/feature_columns/...)에서
    (model, feature_columns) 추출.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    obj = joblib.load(model_path)

    # dict 저장 케이스
    if isinstance(obj, dict):
        model = obj.get("model", obj.get("best_model"))
        if model is None:
            # 첫 키의 값을 모델로 사용 (보통 파이프라인)
            first_key = next(iter(obj.keys()))
            model = obj[first_key]
        feats = obj.get("feature_columns", obj.get("features"))
        if feats is not None:
            return model, list(feats)

        # dict 내 모델 속성에서 feature 추출 시도
        if hasattr(model, "feature_names_in_"):
            return model, list(model.feature_names_in_)
        if hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
            return model, list(model.get_booster().feature_names)
        raise ValueError("모델 dict에서 feature_columns/features를 찾을 수 없습니다.")

    # 단일 모델/파이프라인
    model = obj
    if hasattr(model, "feature_names_in_"):
        return model, list(model.feature_names_in_)
    if hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
        return model, list(model.get_booster().feature_names)

    raise ValueError("모델에서 feature 목록을 추출할 수 없습니다.")


def _predict_attack_proba(model, X: pd.DataFrame) -> np.ndarray:
    """
    모델에서 Attack 확률 벡터(0~1)를 계산.
    - predict_proba[:,1] 우선
    - decision_function 존재 시 시그모이드 변환
    - 그 외에는 predict -> {0,1}
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            p1 = proba[:, 1]
        else:
            # 이진 아닌 케이스: 마지막 칼럼을 공격으로 가정
            p1 = proba[:, -1]
        return np.clip(p1.astype(np.float32), 0.0, 1.0)

    if hasattr(model, "decision_function"):
        z = model.decision_function(X).astype(np.float32)
        return 1.0 / (1.0 + np.exp(-z))

    # fallback
    pred = model.predict(X)
    return np.clip(np.asarray(pred, dtype=np.float32), 0.0, 1.0)


# ---------------------------
# Detection 본체
# ---------------------------
class DetectionTool(ITool):
    """
    입력: rounds_directory 안의 Round_* 입력 CSV
    출력: predictions_directory/{Round}_with_predictions.csv
    - attack_probability, normal_probability
    - predicted_label('Attack'/'Normal'), prediction_confidence
    - (옵션) label이 있으면 is_correct
    """
    def __init__(self,
                 rounds_directory: str = "./test_rounds",
                 predictions_directory: str = "./round_predictions",
                 results_directory: str = "./round_results",  # 남겨둠(호환용, 사용하지 않아도 됨)
                 model_path: Optional[str] = "./models/xgboost_binary_classifier.joblib",
                 rounds: Optional[List[str]] = None,
                 threshold: float = 0.5,
                 force: bool = False):
        self.rounds_directory = rounds_directory
        self.predictions_directory = predictions_directory
        self.results_directory = results_directory
        self.model_path = model_path
        self.rounds = list(rounds) if rounds else None
        self.threshold = float(threshold)
        self.force = bool(force)

    def _choose_rounds(self) -> List[Tuple[str, str]]:
        """라운드 선택/필터 적용."""
        all_pairs = _list_round_inputs(self.rounds_directory)
        if not all_pairs:
            return []
        if self.rounds is None:
            return all_pairs
        want = set(self.rounds)
        return [p for p in all_pairs if p[0] in want]

    def run(self) -> ToolResult:
        ensure_dir(self.predictions_directory)
        # results_directory는 유지용(필수 아님)
        ensure_dir(self.results_directory)

        pairs = self._choose_rounds()
        if not pairs:
            return ToolResult(False, f"입력 라운드 파일을 찾지 못했습니다: {self.rounds_directory}/Round_**.csv")

        # 모델 없으면 안전하게 실패 (무작위 예측 방지)
        if not self.model_path or not os.path.exists(self.model_path):
            return ToolResult(False, f"모델 파일이 없습니다: {self.model_path}")

        try:
            model, feat_cols = _load_model_and_features(self.model_path)
            log.info(f"모델/피처 로드 완료: features={len(feat_cols)}")
        except Exception as e:
            return ToolResult(False, f"모델 로드 실패: {e}")

        produced = []

        for round_name, in_path in pairs:
            out_path = os.path.join(self.predictions_directory, f"{round_name}_with_predictions.csv")
            if os.path.exists(out_path) and not self.force:
                log.info(f"[{round_name}] 기존 예측 존재 → 스킵: {out_path}")
                produced.append(out_path)
                continue

            try:
                df = pd.read_csv(in_path, low_memory=False)
            except Exception as e:
                log.error(f"[{round_name}] 입력 로드 실패: {in_path} → {e}")
                continue

            # feature 교집합
            use_cols = [c for c in feat_cols if c in df.columns]
            if not use_cols:
                log.error(f"[{round_name}] 사용 가능한 feature 없음 (모델 피처 vs 입력 교집합 0)")
                continue

            X = df[use_cols].astype(np.float32)
            p_attack = _predict_attack_proba(model, X)
            p_normal = 1.0 - p_attack

            pred_label = np.where(p_attack >= self.threshold, "Attack", "Normal")
            pred_conf = np.maximum(p_attack, p_normal)

            out = df.copy()
            out["attack_probability"] = p_attack
            out["normal_probability"] = p_normal
            out["predicted_label"] = pred_label
            out["prediction_confidence"] = pred_conf

            if "label" in out.columns:
                out["is_correct"] = (out["label"] == out["predicted_label"])

            try:
                out.to_csv(out_path, index=False, encoding="utf-8")
                log.info(f"[{round_name}] Detection 완료 → {out_path}")
                produced.append(out_path)
            except Exception as e:
                log.error(f"[{round_name}] 저장 실패: {out_path} → {e}")

        if not produced:
            return ToolResult(False, "Detection 수행했지만 어떤 라운드도 출력이 생성되지 않았습니다.")
        return ToolResult(True, "ok", data={"produced": produced})
