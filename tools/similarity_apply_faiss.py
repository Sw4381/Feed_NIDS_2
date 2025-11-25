# -*- coding: utf-8 -*-
"""
Similarity Apply Tool with FAISS (Feedback Base Integration)
KB 미적용 케이스에 대해 FAISS 기반 Feedback 코퍼스 적용
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tools.base import ITool, ToolResult, ensure_dir, get_logger
from tools.feedback_base import FeedbackBase

log = get_logger("SimilarityApplyFAISS")


def _shap_rank_map(row: pd.Series, k: int = 5) -> Dict[str, int]:
    """SHAP Top-K feature → {feature_name: rank}"""
    d = {}
    for i in range(1, k + 1):
        col = f"shap_top{i}_feature"
        f = row.get(col, "")
        if f and str(f) not in ("", "nan", "None"):
            d[str(f)] = i
    return d


def _calculate_similarity_score(
    query_ip_a: str,
    query_ip_b: str,
    query_vec: np.ndarray,
    query_shap: Dict[str, int],
    fb_meta: Dict,
    fb_vec: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    direction_sensitive: bool,
    rank_weights: Dict[int, float],
) -> Tuple[float, Dict]:
    """
    유사도 점수 계산 (IP + Cosine + SHAP Overlap)
    
    Returns:
        (score, components_dict)
    """
    # IP 유사도
    fb_ip_a = str(fb_meta.get("A_ip", ""))
    fb_ip_b = str(fb_meta.get("B_ip", ""))
    
    if direction_sensitive:
        ip_score = 1.0 if (query_ip_a == fb_ip_a and query_ip_b == fb_ip_b) else 0.0
    else:
        ip_score = 1.0 if {query_ip_a, query_ip_b} == {fb_ip_a, fb_ip_b} else 0.0
    
    # Cosine 유사도
    query_norm = np.linalg.norm(query_vec)
    fb_norm = np.linalg.norm(fb_vec)
    
    if query_norm > 0 and fb_norm > 0:
        cos_sim = np.dot(query_vec, fb_vec) / (query_norm * fb_norm)
        cos_score = float((cos_sim + 1) / 2)  # -1~1 → 0~1
        cos_score = np.clip(cos_score, 0.0, 1.0)
    else:
        cos_score = 0.0
    
    # SHAP Overlap
    fb_shap = fb_meta.get("shap_features", {})
    
    if gamma > 0 and query_shap and fb_shap:
        common = set(query_shap.keys()) & set(fb_shap.keys())
        if common:
            overlap_sum = sum(
                rank_weights.get(abs(query_shap[feat] - fb_shap[feat]), 0.0)
                for feat in common
            )
            max_possible = min(len(query_shap), len(fb_shap))
            overlap_score = overlap_sum / max_possible if max_possible > 0 else 0.0
            common_features = sorted(list(common))
        else:
            overlap_score = 0.0
            common_features = []
    else:
        overlap_score = 0.0
        common_features = []
    
    # 최종 점수
    total_score = alpha * ip_score + beta * cos_score + gamma * overlap_score
    
    components = {
        "ip": ip_score,
        "cos": cos_score,
        "overlap": overlap_score,
        "common_features": ",".join(common_features) if common_features else "",
    }
    
    return total_score, components


class SimilarityApplyToolFAISS(ITool):
    """
    FAISS 기반 Feedback 적용 도구
    
    입력: ./kb_applied_round_predictions/{Round}_kb_applied.csv
    동작: kb_applied=False인 케이스만 FeedbackBase FAISS 검색으로 적용
    출력: ./round_predictions_applied/{Round}_position_aware_optimal.csv
    """
    
    def __init__(
        self,
        round_name: str,
        kb_applied_dir: str = "./kb_applied_round_predictions",
        feedback_base: Optional[FeedbackBase] = None,
        out_dir: str = "./round_predictions_applied",
        alpha: float = 0.3,
        beta: float = 0.4,
        gamma: float = 0.3,
        threshold: float = 0.8,
        direction_sensitive: bool = True,
        top_k: int = 5,
        rank_weights: Optional[Dict[int, float]] = None,
        faiss_k: int = 20,
    ):
        """
        Args:
            round_name: 라운드 이름
            kb_applied_dir: KB 적용 결과 디렉토리
            feedback_base: FeedbackBase 인스턴스 (FAISS 포함)
            out_dir: 출력 디렉토리
            alpha: IP 유사도 가중치
            beta: Cosine 유사도 가중치
            gamma: SHAP Overlap 가중치
            threshold: 적용 임계값
            direction_sensitive: 방향 고려 여부
            top_k: SHAP Top-K
            rank_weights: SHAP 순위 가중치
            faiss_k: FAISS Stage 1 후보 개수
        """
        self.round_name = round_name
        self.kb_applied_dir = kb_applied_dir
        self.feedback_base = feedback_base
        self.out_dir = out_dir
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        
        self.direction_sensitive = direction_sensitive
        self.top_k = top_k
        self.rank_weights = rank_weights or {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
        
        self.faiss_k = faiss_k
        
        # FAISS 사용 가능 여부
        self.use_faiss = (
            feedback_base is not None
            and hasattr(feedback_base, "faiss_built")
            and feedback_base.faiss_built
        )

    def run(self) -> ToolResult:
        ensure_dir(self.out_dir)
        
        # FeedbackBase 확인
        if self.feedback_base is None or not self.feedback_base.is_loaded:
            return ToolResult(
                False,
                "FeedbackBase가 로드되지 않았습니다. 파이프라인에서 먼저 로드해주세요.",
            )
        
        # KB 적용 결과 파일 읽기
        kb_applied_file = os.path.join(
            self.kb_applied_dir, f"{self.round_name}_kb_applied.csv"
        )
        if not os.path.exists(kb_applied_file):
            return ToolResult(False, f"KB 적용 파일 없음: {kb_applied_file}")

        df = pd.read_csv(kb_applied_file, low_memory=False)
        
        # KB 필드 확인
        if "kb_applied" not in df.columns:
            df["kb_applied"] = False
        
        # 전체 케이스 수
        total_cases = len(df)
        kb_applied_cases = int(df["kb_applied"].sum())
        kb_unapplied_cases = total_cases - kb_applied_cases
        
        log.info(f"[{self.round_name}] 전체: {total_cases}, KB 적용: {kb_applied_cases}, KB 미적용: {kb_unapplied_cases}")
        
        # KB로 처리되지 않은 케이스만 필터링
        df_unapplied = df[df["kb_applied"] == False].copy()
        
        if df_unapplied.empty:
            log.info(f"[{self.round_name}] 모든 케이스가 KB로 처리됨 → Feedback 스킵")
            out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
            
            # KB 적용된 케이스의 adjusted_label 설정
            df.loc[df["kb_applied"] == True, "adjusted_label"] = df.loc[
                df["kb_applied"] == True, "kb_applied_label"
            ]
            
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(
                True,
                "all_kb_applied",
                output_path=out_path,
                data={"applied": 0, "total": 0, "kb_handled": kb_applied_cases},
            )

        log.info(f"[{self.round_name}] Feedback 처리 대상: {len(df_unapplied)}개")
        
        # Feedback Base 통계
        fb_stats = self.feedback_base.get_stats()
        log.info(f"[{self.round_name}] Feedback Base: {fb_stats.get('total', 0)}개 케이스")
        
        if fb_stats.get('total', 0) == 0:
            log.warning(f"[{self.round_name}] Feedback 코퍼스 없음 → KB 결과만 저장")
            out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
            
            # KB 적용된 케이스의 adjusted_label 설정
            df.loc[df["kb_applied"] == True, "adjusted_label"] = df.loc[
                df["kb_applied"] == True, "kb_applied_label"
            ]
            
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(
                True,
                "no_feedback_corpus",
                output_path=out_path,
                data={"applied": 0, "total": len(df_unapplied), "kb_handled": kb_applied_cases},
            )
        
        # Feature 컬럼 확인
        fb_feature_cols = self.feedback_base.feature_columns
        available_features = [f for f in fb_feature_cols if f in df.columns]
        
        if not available_features:
            log.warning(f"[{self.round_name}] 공통 피처 없음 → KB 결과만 저장")
            out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
            
            # KB 적용된 케이스의 adjusted_label 설정
            df.loc[df["kb_applied"] == True, "adjusted_label"] = df.loc[
                df["kb_applied"] == True, "kb_applied_label"
            ]
            
            df.to_csv(out_path, index=False, encoding="utf-8")
            return ToolResult(
                True,
                "no_common_features",
                output_path=out_path,
                data={"applied": 0, "total": len(df_unapplied), "kb_handled": kb_applied_cases},
            )
        
        log.info(f"[{self.round_name}] 공통 Feature: {len(available_features)}개")
        
        # SHAP 확인
        gamma = self.gamma
        if gamma > 0:
            miss = [
                f"shap_top{i}_feature"
                for i in range(1, self.top_k + 1)
                if f"shap_top{i}_feature" not in df.columns
            ]
            if miss:
                log.warning(f"SHAP 컬럼 누락 → gamma=0으로 계속")
                gamma = 0.0
        
        # 출력 필드 준비
        if "adjusted_label" not in df.columns:
            df["adjusted_label"] = df.get("predicted_label", "")
        
        for c in [
            "feedback_applied",
            "applied_from_case",
            "applied_from_file",
            "applied_reason",
            "applied_confidence",
            "applied_common_features",
        ]:
            if c not in df.columns:
                df[c] = False if c == "feedback_applied" else ""
        
        for c in [
            "applied_similarity_score",
            "applied_similarity_ip",
            "applied_similarity_cosine",
            "applied_similarity_overlap",
        ]:
            if c not in df.columns:
                df[c] = 0.0
        
        # KB 적용된 케이스는 adjusted_label을 kb_applied_label로 설정
        kb_mask = df["kb_applied"] == True
        df.loc[kb_mask, "adjusted_label"] = df.loc[kb_mask, "kb_applied_label"]
        
        # FAISS 모드 확인
        if not self.use_faiss:
            log.warning(f"[{self.round_name}] FAISS 미구축 → Fallback 모드")
            return self._run_fallback(df, df_unapplied, available_features, gamma, kb_applied_cases)
        
        # FAISS 기반 처리
        log.info(f"[{self.round_name}] 🚀 FAISS 모드 시작")
        applied, total_unapplied = self._run_faiss_mode(
            df, df_unapplied, available_features, gamma
        )
        
        # 결과 저장
        out_path = os.path.join(self.out_dir, f"{self.round_name}_position_aware_optimal.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        
        log.info(
            f" [{self.round_name}] 완료 - KB: {kb_applied_cases}, "
            f"Feedback: {applied}/{total_unapplied}, "
            f"총: {kb_applied_cases + applied}/{total_cases}"
        )
        
        return ToolResult(
            True,
            "ok",
            output_path=out_path,
            data={
                "applied": applied,
                "total": total_unapplied,
                "kb_handled": kb_applied_cases,
                "total_cases": total_cases,
                "mode": "FAISS",
            },
        )

    def _run_faiss_mode(
        self,
        df: pd.DataFrame,
        df_unapplied: pd.DataFrame,
        available_features: List[str],
        gamma: float,
    ) -> Tuple[int, int]:
        """
        FAISS 기반 Feedback 적용
        
        Returns:
            (applied_count, total_unapplied)
        """
        applied = 0
        total_unapplied = len(df_unapplied)
        
        # FeedbackBase 벡터 캐시 (재사용)
        fb_vectors_cache = {}
        for meta in self.feedback_base.metadata:
            idx = meta["idx"]
            vec = self.feedback_base.fb_df.iloc[idx][available_features].values.astype(np.float32)
            vec = np.nan_to_num(vec, nan=0.0)
            fb_vectors_cache[idx] = vec
        
        log.info(f"  Feedback 벡터 캐시 구축 완료 ({len(fb_vectors_cache)}개)")
        
        start_time = time.time()
        
        # 배치 처리
        for batch_idx, (idx, row) in enumerate(df_unapplied.iterrows(), 1):
            # 진행률 로깅
            if batch_idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = batch_idx / elapsed if elapsed > 0 else 0
                eta = (total_unapplied - batch_idx) / rate if rate > 0 else 0
                log.info(
                    f"  진행: {batch_idx}/{total_unapplied} ({batch_idx/total_unapplied*100:.1f}%) "
                    f"| {rate:.1f} cases/s | ETA: {eta:.0f}s"
                )
            
            # Query 벡터
            query_vec = row[available_features].values.astype(np.float32)
            query_vec = np.nan_to_num(query_vec, nan=0.0)
            
            # Query SHAP
            query_shap = _shap_rank_map(row, self.top_k) if gamma > 0 else {}
            
            # Query IP
            query_ip_a = str(row.get("A_ip", ""))
            query_ip_b = str(row.get("B_ip", ""))
            
            # Stage 1: FAISS 검색
            try:
                similar_cases = self.feedback_base.search_similar_cases(
                    query_features=query_vec,
                    query_shap_features=query_shap,
                    k=self.faiss_k,
                )
            except Exception as e:
                log.warning(f"  케이스 {idx} FAISS 검색 실패: {e}")
                continue
            
            if not similar_cases:
                continue
            
            # Stage 2: 정확한 유사도 계산
            best_score = 0.0
            best_meta = None
            best_components = None
            
            for fb_idx, faiss_distance, fb_meta in similar_cases:
                # Feedback 벡터
                fb_vec = fb_vectors_cache.get(fb_idx)
                if fb_vec is None:
                    continue
                
                # 유사도 점수 계산
                score, components = _calculate_similarity_score(
                    query_ip_a=query_ip_a,
                    query_ip_b=query_ip_b,
                    query_vec=query_vec,
                    query_shap=query_shap,
                    fb_meta=fb_meta,
                    fb_vec=fb_vec,
                    alpha=self.alpha,
                    beta=self.beta,
                    gamma=gamma,
                    direction_sensitive=self.direction_sensitive,
                    rank_weights=self.rank_weights,
                )
                
                if score > best_score:
                    best_score = score
                    best_meta = fb_meta
                    best_components = components
            
            # Stage 3: 임계값 이상이면 적용
            if best_meta is not None and best_score >= self.threshold:
                df.at[idx, "adjusted_label"] = best_meta.get("feedback_label", "")
                df.at[idx, "feedback_applied"] = True
                df.at[idx, "applied_from_case"] = best_meta.get("case_id", "")
                df.at[idx, "applied_from_file"] = best_meta.get("source_file", "")
                df.at[idx, "applied_reason"] = best_meta.get("feedback_reason", "")
                df.at[idx, "applied_confidence"] = str(best_meta.get("feedback_confidence", ""))
                df.at[idx, "applied_similarity_score"] = best_score
                df.at[idx, "applied_similarity_ip"] = best_components["ip"]
                df.at[idx, "applied_similarity_cosine"] = best_components["cos"]
                df.at[idx, "applied_similarity_overlap"] = best_components["overlap"]
                df.at[idx, "applied_common_features"] = best_components["common_features"]
                applied += 1
        
        total_elapsed = time.time() - start_time
        rate = total_unapplied / total_elapsed if total_elapsed > 0 else 0
        log.info(f"  ✅ FAISS 처리 완료: {total_elapsed:.2f}초 ({rate:.1f} cases/s)")
        
        return applied, total_unapplied