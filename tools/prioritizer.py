# -*- coding: utf-8 -*-
from __future__ import annotations
import os, numpy as np, pandas as pd
from typing import Optional, List, Tuple
from .base import ITool, ToolResult, ensure_dir, get_logger

log = get_logger("PrioritizerTool")

def _statistical_flags(flow_count: int, unique_ports: int) -> Tuple[int, int]:
    dos_flag = int(flow_count >= 50 and unique_ports < 2)
    portscan_flag = int(unique_ports >= 50 and flow_count >= 50)
    return dos_flag, portscan_flag

def _statistical_score(flow_count: int, unique_ports: int) -> float:
    d, p = _statistical_flags(flow_count, unique_ports)
    return (d + p) / 2.0  # {0.0, 0.5, 1.0}

def _pair_window_stats_directional_from_full_df(df: pd.DataFrame) -> pd.DataFrame:
    need = ["capsule_duration", "A_ip", "B_ip", "B_port"]
    tmp = df.copy()
    for c in need:
        if c not in tmp.columns:
            tmp[c] = np.nan
    tmp["_kc"] = tmp["capsule_duration"].astype(str)
    tmp["_ka"] = tmp["A_ip"].astype(str)
    tmp["_kb"] = tmp["B_ip"].astype(str)
    g = tmp.groupby(["_kc", "_ka", "_kb"], dropna=False)
    stats = g.agg(flow_count=("B_port", "size"),
                  unique_ports=("B_port", lambda s: s.nunique(dropna=False))).reset_index()
    return stats.rename(columns={"_kc":"capsule_duration","_ka":"A_ip","_kb":"B_ip"})

class PrioritizerTool(ITool):
    """
    입력: ./round_predictions/{Round}_with_predictions.csv
    출력: ./feedback_cases/{Round}_low_confidence_cases.csv
    """
    def __init__(self,
                 round_name: str,
                 pred_dir: str = "./round_predictions",
                 out_dir: str = "./feedback_cases",
                 alpha: float = 0.3,
                 beta: float  = 0.7,
                 bottom_percent: Optional[float] = 5.0,
                 top_k: Optional[int] = None,
                 model_path: Optional[str] = "./models/xgboost_binary_classifier.joblib",
                 enable_shap: bool = True):
        self.round_name = round_name
        self.pred_dir = pred_dir
        self.out_dir = out_dir
        self.alpha = alpha
        self.beta = beta
        self.bottom_percent = bottom_percent
        self.top_k = top_k
        self.model_path = model_path
        self.enable_shap = enable_shap

    def _load_model_and_features(self):
        if not self.enable_shap:
            return None, []
        if not self.model_path or not os.path.exists(self.model_path):
            log.warning("모델 파일 없음 → SHAP 생략")
            return None, []
        import joblib
        obj = joblib.load(self.model_path)
        if isinstance(obj, dict):
            model = obj.get("model") or obj.get("best_model") or list(obj.values())[0]
            feats = obj.get("feature_columns") or obj.get("features") or []
        else:
            model = obj
            if hasattr(model, "feature_names_in_"):
                feats = list(model.feature_names_in_)
            elif hasattr(model, "get_booster") and hasattr(model.get_booster(), "feature_names"):
                feats = list(model.get_booster().feature_names)  # XGBoost
            else:
                feats = []
        return model, feats

    def _calc_shap(self, model, X: pd.DataFrame):
        try:
            import shap
            if hasattr(model, 'get_booster') or hasattr(model, 'estimators_'):
                exp = shap.TreeExplainer(model)
            else:
                bg = shap.sample(X, min(100, len(X)))
                exp = shap.KernelExplainer(model.predict_proba, bg)
            vals = exp.shap_values(X)
            if isinstance(vals, list):  # binary → class-1
                vals = vals[1]
            return vals
        except Exception as e:
            log.warning(f"SHAP 계산 실패: {e}")
            return None

    def _add_shap_columns(self, df_targets: pd.DataFrame, shap_values, feat_cols: List[str]) -> pd.DataFrame:
        if shap_values is None or len(feat_cols) == 0:
            return df_targets
        out = df_targets.copy()
        abs_shap = np.abs(shap_values)
        idx_top5 = np.argsort(abs_shap, axis=1)[:, -5:][:, ::-1]
        for r in range(1, 6):
            fn, fv, sv = [], [], []
            for i, idxs in enumerate(idx_top5):
                j = idxs[r-1]
                fn.append(feat_cols[j])
                sv.append(float(abs_shap[i, j]))
                fv.append(df_targets.iloc[i][feat_cols[j]] if feat_cols[j] in df_targets.columns else np.nan)
            out[f"shap_top{r}_feature"] = fn
            out[f"shap_top{r}_value"] = sv
            out[f"shap_top{r}_feature_value"] = fv
        out["shap_total_importance"] = abs_shap.sum(axis=1)
        return out

    def run(self) -> ToolResult:
        ensure_dir(self.out_dir)
        src = os.path.join(self.pred_dir, f"{self.round_name}_with_predictions.csv")
        if not os.path.exists(src):
            return ToolResult(False, f"예측 파일 없음: {src}")

        df = pd.read_csv(src, low_memory=False)
        req = ["predicted_label","attack_probability","prediction_confidence","label",
               "normal_probability","is_correct","capsule_duration","A_ip","B_ip","B_port"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            return ToolResult(False, f"필수 컬럼 누락: {miss}")

        att = df[df["predicted_label"] == "Attack"].copy()
        if att.empty:
            return ToolResult(True, "Attack 없음 → 스킵", output_path=None)

        stats = _pair_window_stats_directional_from_full_df(df)
        att = att.merge(stats, on=["capsule_duration","A_ip","B_ip"], how="left", validate="m:1")
        att["flow_count"]   = pd.to_numeric(att["flow_count"], errors="coerce").fillna(0).astype(int)
        att["unique_ports"] = pd.to_numeric(att["unique_ports"], errors="coerce").fillna(0).astype(int)
        att["statistical_score"] = att.apply(lambda r: _statistical_score(int(r["flow_count"]), int(r["unique_ports"])), axis=1)

        att["attack_probability"] = pd.to_numeric(att["attack_probability"], errors="coerce").fillna(0.0)
        att["statistical_score"]  = pd.to_numeric(att["statistical_score"],  errors="coerce").fillna(0.0)
        att["confidence_score"]   = self.alpha*att["attack_probability"] + self.beta*att["statistical_score"]
        att = att.sort_values(["confidence_score","attack_probability"], ascending=[True,True], kind="mergesort")

        if self.top_k is not None and self.bottom_percent is not None:
            return ToolResult(False, "top_k와 bottom_percent 동시 지정 불가")

        if self.bottom_percent is not None:
            if not (0 < self.bottom_percent <= 100):
                return ToolResult(False, "bottom_percent는 0~100")
            k = max(1, int(len(att) * (self.bottom_percent/100.0)))
        elif self.top_k is not None:
            k = min(int(self.top_k), len(att))
        else:
            return ToolResult(False, "top_k 또는 bottom_percent 지정 필요")

        targets = att.head(k).copy()

        # SHAP
        model, feat_cols = self._load_model_and_features()
        if model is not None and feat_cols:
            avail = [c for c in feat_cols if c in targets.columns]
            if avail:
                vals = self._calc_shap(model, targets[avail])
                targets = self._add_shap_columns(targets, vals, avail)

        # 출력(원본 + 통계 + SHAP + 피드백 입력 필드)
        original_cols = list(df.columns)
        out_df = targets[original_cols].copy()
        for c in ["flow_count","unique_ports","statistical_score","confidence_score"]:
            if c in targets.columns: out_df[c] = targets[c].values
        for c in [c for c in targets.columns if c.startswith("shap_")]:
            out_df[c] = targets[c].values
        for c, default in [
            ("feedback_label",""),("feedback_confidence",""),("feedback_reason",""),
            ("needs_review",True),("reviewed",False),("review_date",""),
        ]:
            if c not in out_df.columns: out_df[c] = default

        out_df = out_df.reset_index(drop=True)
        out_df.insert(0, "case_id", [f"{self.round_name}_lowconf_{i+1:04d}" for i in range(len(out_df))])

        out_path = os.path.join(self.out_dir, f"{self.round_name}_low_confidence_cases.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"선택 {len(out_df)}/{len(att)} → {out_path}")
        return ToolResult(True, "ok", output_path=out_path, data={"selected": len(out_df), "attack_count": len(att)})
