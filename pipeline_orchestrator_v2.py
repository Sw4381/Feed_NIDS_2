#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FEED-NIDS Pipeline Orchestrator v2 (FAISS 통합 + 동적 Feedback 로드)
실행 흐름:
  Detection → Prioritizer → Train KB (FAISS) → Feedback (FAISS, 동적 증축) → 최종 결과
"""

import sys
if sys.version_info < (3, 8):
    raise SystemExit("Python 3.8+ 필요합니다.")

import os
import glob
import argparse
import pandas as pd
import numpy as np

from tools.base import get_logger, ToolResult
from tools.detection import DetectionTool
from tools.prioritizer import PrioritizerTool
from tools.auto_feedback import AutoFeedbackTool
from tools.kb_similarity_apply_tool_optimized import KBSimilarityApplyToolOptimized
from tools.knowledge_base import KnowledgeBase
from tools.feedback_base import FeedbackBase
from tools.similarity_apply_faiss import SimilarityApplyToolFAISS
from tools.merge import MergeTool


log = get_logger("Orchestrator-v2")


def list_rounds_from_predictions(pred_dir: str):
    """round_predictions에서 Round 목록 산출"""
    paths = sorted(glob.glob(os.path.join(pred_dir, "*_with_predictions.csv")))
    return [os.path.basename(p).replace("_with_predictions.csv", "") for p in paths]


def list_rounds_from_inputs(det_in_dir: str):
    """입력 폴더에서 Round 목록 산출"""
    names = set()
    for p in glob.glob(os.path.join(det_in_dir, "Round_*.csv")):
        bn = os.path.basename(p)
        rn = bn[:-4]
        if rn.endswith("_raw"):
            rn = rn[:-4]
        if rn.startswith("Round_"):
            names.add(rn)
    return sorted(names)


def main():
    ap = argparse.ArgumentParser(
        description="FEED-NIDS v2: Detection → Prioritizer → Train KB (FAISS) → Feedback (FAISS)"
    )
    
    # 실행 모드
    ap.add_argument(
        "--mode", 
        choices=["kb-only", "feedback-only", "full"],
        default="full",
        help="kb-only: Train KB만, feedback-only: Feedback만, full: 전체(권장)"
    )
    
    # 라운드 선택
    ap.add_argument("--rounds", nargs="*", default=None, help="예: Round_1 Round_3")
    ap.add_argument("--all", action="store_true", help="모든 라운드 처리")
    
    # 디렉토리 경로
    ap.add_argument("--det-in", default="./test_rounds", help="Detection 입력")
    ap.add_argument("--pred-dir", default="./round_predictions", help="예측 결과")
    ap.add_argument("--feedback-dir", default="./feedback_cases", help="피드백 케이스")
    ap.add_argument("--applied-dir", default="./round_predictions_applied", help="적용 결과")
    ap.add_argument("--train-cases-dir", default="./tools/Train_Cases", help="Train KB 위치")
    ap.add_argument("--model-path", default="./models/xgboost_binary_classifier.joblib")
    ap.add_argument("--det-out", default="./round_results")
    ap.add_argument("--kb-applied-dir", default="./kb_applied_round_predictions", help="KB 적용 결과")

    # Detection (0단계)
    ap.add_argument("--skip-detection", action="store_true", help="Detection 스킵")
    ap.add_argument("--force-detection", action="store_true", help="Detection 재실행")
    ap.add_argument("--det-threshold", type=float, default=0.5)

    # Phase 1: Train KB 파라미터
    ap.add_argument("--kb-alpha", type=float, default=0.3, help="KB IP 가중치")
    ap.add_argument("--kb-beta", type=float, default=0.4, help="KB Cosine 가중치")
    ap.add_argument("--kb-gamma", type=float, default=0.3, help="KB SHAP 가중치")
    ap.add_argument("--kb-threshold", type=float, default=0.9, help="KB 유사도 임계값")
    ap.add_argument("--kb-no-direction", action="store_true", help="KB 방향 무시")
    ap.add_argument("--kb-top-k", type=int, default=5, help="KB SHAP Top-K")
    ap.add_argument("--kb-no-faiss", action="store_true", help="FAISS 비활성화 (기본 Brute-Force 사용)")
    ap.add_argument("--kb-faiss-k", type=int, default=500, help="FAISS Stage 1 후보 개수")
    ap.add_argument("--kb-cache-dir", default="./cache", help="FAISS 캐시 디렉토리")
    ap.add_argument("--kb-use-gpu", action="store_true", help="KB FAISS GPU 사용")
    
    # Phase 2: Gating 파라미터
    ap.add_argument("--gate-alpha", type=float, default=0.3, help="Gating: 공격 확률")
    ap.add_argument("--gate-beta", type=float, default=0.7, help="Gating: 통계 점수")
    ap.add_argument("--gate-bottom-percent", type=float, default=5.0, help="Gating: 하위%")
    ap.add_argument("--gate-top-k", type=int, default=None, help="Gating: 고정 K")
    ap.add_argument("--gate-no-shap", action="store_true", help="Gating: SHAP 비활성화")

    # Phase 2: Auto-feedback
    ap.add_argument("--skip-auto-feedback", action="store_true")
    ap.add_argument("--auto-top-n", type=int, default=300)
    ap.add_argument("--auto-percent", type=float, default=None)

    # Phase 3: Feedback Base (FAISS)
    ap.add_argument("--fb-no-faiss", action="store_true", help="Feedback FAISS 비활성화")
    ap.add_argument("--fb-cache-dir", default="./cache", help="Feedback FAISS 캐시 디렉토리")
    ap.add_argument("--fb-use-gpu", action="store_true", help="Feedback FAISS GPU 사용")
    ap.add_argument("--fb-no-auto-rebuild", action="store_true", help="자동 재구축 비활성화")

    # Phase 3: Similarity Apply (Feedback)
    ap.add_argument("--alpha", type=float, default=0.3, help="Feedback IP 가중치")
    ap.add_argument("--beta", type=float, default=0.4, help="Feedback Cosine 가중치")
    ap.add_argument("--gamma", type=float, default=0.3, help="Feedback SHAP 가중치")
    ap.add_argument("--threshold", type=float, default=0.8, help="Feedback 임계값")
    ap.add_argument("--no-direction", action="store_true")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--faiss-k", type=int, default=900, help="Feedback FAISS Stage 1 후보")

    # 라운드별 스킵
    ap.add_argument("--skip-feedback-rounds", nargs="*", default=[])
    ap.add_argument("--skip-feedback-round1", action="store_true")

    args = ap.parse_args()

    # ===== 라운드 결정 =====
    if args.rounds:
        rounds = args.rounds
    elif args.all:
        rounds = list_rounds_from_predictions(args.pred_dir)
        if not rounds:
            rounds = list_rounds_from_inputs(args.det_in)
    else:
        log.error("처리할 라운드 없음. (--rounds ... 또는 --all)")
        return

    if not rounds:
        log.error("라운드 목록 없음")
        return

    log.info("=" * 60)
    log.info(f"처리 라운드: {rounds}")
    log.info(f"실행 모드: {args.mode} (kb-only/feedback-only/full)")
    log.info("=" * 60)

    # ===== Phase 0: Detection =====
    if not args.skip_detection:
        log.info("Phase 0️⃣: Detection 시작")
        det = DetectionTool(
            rounds_directory=args.det_in,
            predictions_directory=args.pred_dir,
            results_directory=args.det_out,
            model_path=args.model_path,
            rounds=rounds,
            threshold=args.det_threshold,
            force=args.force_detection
        ).run()
        if not det.ok:
            log.error(f"Detection 실패: {det.message}")
            return
        log.info("✅ Phase 0: Detection 완료")
    else:
        log.info("⭐️ Phase 0: Detection 스킵")

    log.info("")

    # ===== Phase 1️⃣: Prioritizer (분석 대상 추출) - 항상 실행 =====
    log.info("Phase 1️⃣: Prioritizer (분석 대상 추출)")
    log.info("-" * 60)

    prioritizer_results = {}
    for rn in rounds:
        log.info(f"[{rn}] Prioritizer 시작")
        pr = PrioritizerTool(
            round_name=rn,
            pred_dir=args.pred_dir,
            out_dir=args.feedback_dir,
            alpha=args.gate_alpha,
            beta=args.gate_beta,
            bottom_percent=args.gate_bottom_percent if args.gate_top_k is None else None,
            top_k=args.gate_top_k,
            model_path=args.model_path,
            enable_shap=not args.gate_no_shap,
        ).run()
        
        prioritizer_results[rn] = pr
        if pr.ok:
            log.info(f"✅ [{rn}] Prioritizer 완료: {pr.data}")
        else:
            log.warning(f"⚠️ [{rn}] Prioritizer: {pr.message}")

    log.info("✅ Phase 1: Prioritizer 완료")
    log.info("")

    # ===== Phase 2️⃣: Train Knowledge Base 적용 (선택사항) =====
    if args.mode in ["full", "kb-only"]:
        log.info("Phase 2️⃣: Train Knowledge Base 적용")
        log.info("-" * 60)

        # KB 로드 (FAISS 자동 구축)
        kb = KnowledgeBase(
            train_cases_dir=args.train_cases_dir,
            use_faiss=not args.kb_no_faiss,
            cache_dir=args.kb_cache_dir,
            index_type="IVF",
            n_clusters=100,
            use_gpu=args.kb_use_gpu
        )

        if kb.load():
            stats = kb.get_stats()
            log.info(f"✅ Knowledge Base 로드: {stats}")
            kb_corpus = kb.kb_df.copy()
        else:
            log.warning("⚠️ Knowledge Base 로드 실패 → Phase 2 스킵")
            kb_corpus = None
            kb = None

        # 모든 라운드에 KB 적용 (FAISS 최적화)
        kb_results = {}
        if kb_corpus is not None:
            for rn in rounds:
                log.info(f"[{rn}] KB 적용 시작")
                
                kb_tool = KBSimilarityApplyToolOptimized(
                    round_name=rn,
                    pred_dir=args.feedback_dir,
                    kb_corpus=kb.kb_df.copy(),
                    kb_instance=kb,
                    out_dir=args.kb_applied_dir,
                    alpha=args.kb_alpha,
                    beta=args.kb_beta,
                    gamma=args.kb_gamma,
                    threshold=args.kb_threshold,
                    direction_sensitive=not args.kb_no_direction,
                    top_k=args.kb_top_k,
                    faiss_k=args.kb_faiss_k,
                ).run()
            
                kb_results[rn] = kb_tool
                if kb_tool.ok:
                    mode = kb_tool.data.get("mode", "Unknown")
                    applied = kb_tool.data.get("applied", 0)
                    total = kb_tool.data.get("total", 0)
                    log.info(f"✅ [{rn}] KB 적용 완료 ({mode}): {applied}/{total}")
                else:
                    log.warning(f"⚠️ [{rn}] KB: {kb_tool.message}")

        log.info("✅ Phase 2: Train KB 완료")
        log.info("")

        # kb-only 모드면 여기서 종료
        if args.mode == "kb-only":
            log.info("=" * 60)
            log.info("🎉 kb-only 모드 완료!")
            log.info("=" * 60)
            return
    else:
        log.info("⭐️ Phase 2: Train KB 스킵 (--mode feedback-only)")
        log.info("")
        kb = None

    # ===== Phase 3️⃣: Feedback Base 로드 및 적용 (선택사항, 동적 로드) =====
    if args.mode in ["full", "feedback-only"]:
        log.info("Phase 3️⃣: Feedback Base 로드 및 적용 (동적 증축)")
        log.info("-" * 60)

        if args.skip_feedback_round1:
            args.skip_feedback_rounds = list(set(args.skip_feedback_rounds + ["Round_1"]))
        skip_set = set(args.skip_feedback_rounds)

        # 🔥 Feedback Base를 각 라운드마다 동적으로 로드
        feedback_base = None
        
        for rn in rounds:
            log.info(f"[{rn}] Feedback 처리 시작")

            # 자동 피드백
            if not args.skip_auto_feedback and prioritizer_results.get(rn) and prioritizer_results[rn].ok:
                log.info(f"  └─ AutoFeedback")
                af = AutoFeedbackTool(
                    round_name=rn,
                    feedback_dir=args.feedback_dir,
                    top_n=args.auto_top_n,
                    percent=args.auto_percent,
                ).run()
                if af.ok:
                    log.info(f"    ✅ AutoFeedback 완료")
                    
                    # 🔥 AutoFeedback 후 FeedbackBase 재로드 (자동 증축)
                    log.info(f"  └─ FeedbackBase 재로드 (AutoFeedback 반영)")
                    feedback_base = FeedbackBase(
                        feedback_dir=args.feedback_dir,
                        use_faiss=not args.fb_no_faiss,
                        cache_dir=args.fb_cache_dir,
                        index_type="IVF",
                        n_clusters=100,
                        use_gpu=args.fb_use_gpu,
                        auto_rebuild=not args.fb_no_auto_rebuild,
                    )
                    
                    if feedback_base.load(force_rebuild=True):
                        fb_stats = feedback_base.get_stats()
                        log.info(f"    ✅ Feedback Base 재로드 완료: {fb_stats}")
                    else:
                        log.warning(f"    ⚠️ Feedback Base 재로드 실패")
                        feedback_base = None
                else:
                    log.warning(f"    ⚠️ AutoFeedback: {af.message}")

            # 🔥 Round_1은 피드백 적용 스킵 (코퍼스 구축 단계)
            if rn == "Round_1":
                log.info(f"  └─ Round_1: 피드백 코퍼스 구축 단계 → SimilarityApply 스킵")
                log.info(f"✅ [{rn}] Feedback 처리 완료 (코퍼스 구축)")
                continue

            # 유사도 적용 (FAISS)
            if rn in skip_set:
                log.info(f"  └─ SimilarityApply 스킵")
            else:
                log.info(f"  └─ SimilarityApply (FAISS)")
                
                if feedback_base is None:
                    log.warning(f"    ⚠️ Feedback Base 없음 → 스킵")
                    continue
                
                sa = SimilarityApplyToolFAISS(
                    round_name=rn,
                    kb_applied_dir=args.kb_applied_dir,
                    feedback_base=feedback_base,
                    out_dir=args.applied_dir,
                    alpha=args.alpha,
                    beta=args.beta,
                    gamma=args.gamma,
                    threshold=args.threshold,
                    direction_sensitive=not args.no_direction,
                    top_k=args.top_k,
                    faiss_k=args.faiss_k,
                ).run()
                
                if sa.ok:
                    log.info(f"    ✅ SimilarityApply 완료: {sa.data}")
                else:
                    log.error(f"    ❌ SimilarityApply 실패: {sa.message}")
                    continue

            # Merge
            log.info(f"  └─ Merge")
            mg = MergeTool(
                round_name=rn,
                pred_dir=args.pred_dir,
                applied_dir=args.applied_dir,
            ).run()
            
            if mg.ok:
                log.info(f"    ✅ Merge 완료")
            else:
                log.error(f"    ❌ Merge 실패: {mg.message}")

            log.info(f"✅ [{rn}] Feedback 처리 완료")

        log.info("✅ Phase 3: Feedback 완료")
        log.info("")

    else:
        log.info("⭐️ Phase 3: Feedback 스킵 (--mode kb-only)")
        log.info("")

    log.info("=" * 60)
    log.info("🎉 전체 파이프라인 완료!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()