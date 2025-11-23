# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re
from typing import Dict, List, Tuple, Optional
import numpy as np, pandas as pd
from .base import ITool, ToolResult, ensure_dir, get_logger

log = get_logger("SimilarityApplyTool")

def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    patterns = [
        r'^label$', r'.*_label$', r'^case_id$', r'^capsule_id$',
        r'.*feedback.*', r'.*applied.*', r'^shap_.*', r'.*reviewed.*',
        r'.*needs_review.*', r'.*review_date.*', r'^A_ip$', r'^B_ip$',
        r'^A_port$', r'^B_port$', r'.*predicted.*', r'.*confidence.*',
        r'.*probability.*', r'.*is_correct.*', r'.*adjusted.*', r'.*reason.*',
        r'.*round.*', r'^capsule_duration$', r'^flow_count$', r'^unique_ports$',
        r'^statistical_score$', r'^__.*',
    ]
    comp = [re.compile(p, re.IGNORECASE) for p in patterns]
    num = df.select_dtypes(include=['int64','float64']).columns
    feats = [c for c in num if not any(p.match(c) for p in comp)]
    return feats

def _to_vec(row: pd.Series, feat_cols: List[str]) -> np.ndarray:
    return np.nan_to_num(row[feat_cols].values, nan=0.0).astype(np.float64)

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
    if n1==0 or n2==0: return 0.0
    v = float(np.dot(a,b)/(n1*n2))
    return max(0.0, min(1.0, (v+1)/2))

def _shap_rank_map(row: pd.Series, k: int=5) -> Dict[str,int]:
    d = {}
    for i in range(1, k+1):
        col = f"shap_top{i}_feature"
        f = row.get(col, "")
        if f and str(f) not in ("","nan"):
            d[str(f)] = i
    return d

def _rank_overlap(t: Dict[str,int], f: Dict[str,int], weights: Dict[int,float]) -> Tuple[float,int,List[str]]:
    if not t or not f: return 0.0, 0, []
    com = set(t) & set(f)
    if not com: return 0.0, 0, []
    s = 0.0
    for feat in com:
        d = abs(t[feat]-f[feat])
        s += weights.get(d, 0.0)
    mx = min(len(t), len(f))
    return (s/mx if mx>0 else 0.0), len(com), sorted(list(com))

def _load_feedback_corpus(feedback_dir: str, labels=("Normal","Attack")) -> pd.DataFrame:
    import glob
    paths = sorted(glob.glob(os.path.join(feedback_dir, "*_low_confidence_cases.csv")))
    outs = []
    for p in paths:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        if "reviewed" not in df.columns or "feedback_label" not in df.columns:
            continue
        sub = df[(df["reviewed"]==True) & (df["feedback_label"].isin(labels))].copy()
        if sub.empty: continue
        if "feedback_confidence" not in sub.columns: sub["feedback_confidence"]=0
        if "review_date" not in sub.columns: sub["review_date"]=""
        sub["__fb_source"] = os.path.basename(p)
        outs.append(sub)
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()

class SimilarityApplyTool(ITool):
    """
    입력: ./feedback_cases/{Round}_low_confidence_cases.csv
    출력: ./round_predictions_applied/{Round}_position_aware_optimal.csv (+ details)
    """
    def __init__(self,
                 round_name: str,
                 feedback_dir: str = "./feedback_cases",
                 out_dir: str = "./round_predictions_applied",
                 alpha: float = 0.5, beta: float = 0.0, gamma: float = 0.5,
                 threshold: float = 0.9, direction_sensitive: bool = True,
                 top_k: int = 5,
                 rank_weights: Optional[Dict[int,float]] = None,
                 restrict_to_attack_predictions: bool = False,
                 chunk_size: int = 1000):
        self.round_name = round_name
        self.feedback_dir = feedback_dir
        self.out_dir = out_dir
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.direction_sensitive = direction_sensitive
        self.top_k = top_k
        self.rank_weights = rank_weights or {0:1.0, 1:0.8, 2:0.6, 3:0.4}
        self.restrict = restrict_to_attack_predictions
        self.chunk_size = chunk_size

    def run(self) -> ToolResult:
        ensure_dir(self.out_dir)
        pred_file = os.path.join(self.feedback_dir, f"{self.round_name}_low_confidence_cases.csv")
        if not os.path.exists(pred_file):
            return ToolResult(False, f"타깃 파일 없음: {pred_file}")

        df = pd.read_csv(pred_file, low_memory=False)
        feat_cols = _get_feature_columns(df)
        gamma = self.gamma
        if gamma > 0:
            miss = [f"shap_top{i}_feature" for i in range(1, self.top_k+1) if f"shap_top{i}_feature" not in df.columns]
            if miss:
                log.warning(f"SHAP 컬럼 누락 → gamma=0으로 계속: {miss}")
                gamma = 0.0

        fb = _load_feedback_corpus(self.feedback_dir, ("Normal","Attack"))
        if fb.empty:
            out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(True, "코퍼스 없음 → 원본 저장", output_path=out_path)

        fb_cols = _get_feature_columns(fb)
        common = sorted(list(set(feat_cols) & set(fb_cols)))
        if not common:
            out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(True, "공통 피처 없음 → 원본 저장", output_path=out_path)

        # 전처리
        fb["__feature_vector"] = fb[common].apply(lambda r: _to_vec(r, common), axis=1)
        fb["__shap_features"]  = fb.apply(lambda r: _shap_rank_map(r, self.top_k), axis=1)

        # 출력 필드 준비
        if "adjusted_label" not in df.columns:
            df["adjusted_label"] = df.get("predicted_label","")
        for c in ["feedback_applied","applied_from_case","applied_from_file",
                  "applied_reason","applied_confidence","applied_common_features"]:
            if c not in df.columns:
                df[c] = False if c=="feedback_applied" else ""
        for c in ["applied_similarity_score","applied_similarity_ip","applied_similarity_cosine","applied_similarity_overlap"]:
            if c not in df.columns: df[c] = 0.0

        # 본 처리
        recs = []
        applied = 0
        total = len(df)
        for s in range(0, total, self.chunk_size):
            e = min(s+self.chunk_size, total)
            log.info(f"{self.round_name} 진행 {e}/{total} ({e/total*100:.1f}%)")
            for idx, row in df.iloc[s:e].iterrows():
                if self.restrict and str(row.get("predicted_label",""))!="Attack":
                    continue
                t_vec = _to_vec(row, common)
                t_shap = _shap_rank_map(row, self.top_k) if gamma>0 else {}
                ta, tb = str(row.get("A_ip","")), str(row.get("B_ip",""))

                best_score, best_row, comp = 0.0, None, None
                for _, fbr in fb.iterrows():
                    if self.direction_sensitive:
                        ip = 1.0 if (ta==str(fbr.get("A_ip","")) and tb==str(fbr.get("B_ip",""))) else 0.0
                    else:
                        ip = 1.0 if {ta,tb} == {str(fbr.get("A_ip","")), str(fbr.get("B_ip",""))} else 0.0
                    cs = _cos(t_vec, fbr["__feature_vector"])
                    if gamma>0:
                        ov, cc, com = _rank_overlap(t_shap, fbr["__shap_features"], self.rank_weights)
                    else:
                        ov, cc, com = 0.0, 0, []
                    score = self.alpha*ip + self.beta*cs + gamma*ov
                    if score > best_score:
                        best_score, best_row = score, fbr
                        comp = {"ip":ip,"cos":cs,"ov":ov,"cc":cc,"com":",".join(com) if com else ""}

                if best_row is not None and best_score >= self.threshold:
                    df.at[idx, "adjusted_label"] = best_row.get("feedback_label","")
                    df.at[idx, "feedback_applied"] = True
                    df.at[idx, "applied_from_case"] = str(best_row.get("case_id",""))
                    df.at[idx, "applied_from_file"] = str(best_row.get("__fb_source",""))
                    df.at[idx, "applied_reason"] = str(best_row.get("feedback_reason",""))
                    df.at[idx, "applied_confidence"] = str(best_row.get("feedback_confidence",""))
                    df.at[idx, "applied_similarity_score"] = best_score
                    df.at[idx, "applied_similarity_ip"] = comp["ip"]
                    df.at[idx, "applied_similarity_cosine"] = comp["cos"]
                    df.at[idx, "applied_similarity_overlap"] = comp["ov"]
                    df.at[idx, "applied_common_features"] = comp["com"]
                    applied += 1

                # 상세 기록(원하면 여기 리스트에 append하여 별도 CSV로 저장 가능)

        out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"유사도 적용 {applied}/{len(df)} → {out_path}")
        return ToolResult(True, "ok", output_path=out_path, data={"applied": applied, "total": len(df)})
    


