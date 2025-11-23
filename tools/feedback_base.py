# -*- coding: utf-8 -*-
"""
Feedback Base Manager with FAISS Vector Store Integration
Reviewed Feedback 케이스들을 로드하고 FAISS 벡터 인덱스로 관리하는 모듈
- 지속적 증축 (reviewed=True 케이스 누적)
- 텍스트 메타데이터 저장 (feedback_reason 등)
- LLM 프롬프트 생성 지원
"""
from __future__ import annotations
import os
import glob
import pickle
import hashlib
import time
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from tools.base import get_logger

log = get_logger("FeedbackBase")

# 네트워크 보안 Feature 22개 (KB와 동일)
NETWORK_SECURITY_FEATURES = [
    'alpahbet_cnt_payload_sum',
    'alpha_cnt_dns_query_sum',
    'client_extensions_cnt',
    'entropys_avg',
    'flow_delta_times_sum',
    'flow_duration_seconds',
    'flow_stdev_time',
    'nonascii_cnt_dns_query_sum',
    'nonascii_cnt_payload_sum',
    'number_cnt_dns_query_sum',
    'number_cnt_payload_sum',
    'payload_len_max',
    'payload_len_min',
    'payload_lens_sum',
    'payload_packets_cnt',
    'query_response_ttls_sum',
    'server_certificates_cnt',
    'server_extensions_cnt',
    'special_cnt_dns_query_sum',
    'special_cnt_payload_sum',
    'tls_SAN_cnt',
    'total_packets_cnt'
]

# FAISS 가용성 체크
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    log.warning("⚠️ FAISS 미설치 - 벡터 인덱스 기능 비활성화 (pip install faiss-cpu)")

# StandardScaler
try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    log.warning("⚠️ scikit-learn 미설치 - 정규화 없이 진행 (pip install scikit-learn)")


class FeedbackBase:
    """
    Feedback 코퍼스 관리 (FAISS + 텍스트 메타데이터)
    
    Features:
    - reviewed=True 케이스만 자동 로드 및 증축
    - FAISS 벡터 인덱스 자동 구축 (22개 Feature + SHAP Top-5)
    - 텍스트 메타데이터 보존 (feedback_reason, feedback_label 등)
    - 캐시 시스템 (hash 기반, 자동 재구축)
    - LLM 프롬프트 생성 지원
    - GPU 지원 (선택 사항)
    """
    
    def __init__(self, 
                 feedback_dir: str = "./feedback_cases",
                 use_faiss: bool = True,
                 cache_dir: str = "./cache",
                 index_type: str = "IVF",
                 n_clusters: int = 100,
                 use_gpu: bool = False,
                 auto_rebuild: bool = True):
        """
        Args:
            feedback_dir: Feedback Cases 디렉토리 경로
            use_faiss: FAISS 벡터 인덱스 사용 여부
            cache_dir: 캐시 디렉토리
            index_type: FAISS 인덱스 타입 ("Flat", "IVF", "HNSW")
            n_clusters: IVF 클러스터 수
            use_gpu: GPU 사용 여부
            auto_rebuild: 새 피드백 감지시 자동 재구축
        """
        self.feedback_dir = feedback_dir
        self.fb_df = None
        self.is_loaded = False
        
        # FAISS 설정
        self.use_faiss = use_faiss and HAS_FAISS
        self.cache_dir = cache_dir
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu and HAS_FAISS
        self.auto_rebuild = auto_rebuild
        
        # FAISS 벡터 저장소
        self.faiss_index = None
        self.scaler = None
        self.feature_columns = []
        self.shap_feature_vocab = []
        self.metadata = []
        self.faiss_built = False
        
        # GPU 리소스 (재사용)
        self.gpu_resources = None
        
        if self.use_faiss:
            os.makedirs(cache_dir, exist_ok=True)

    def load(self, force_rebuild: bool = False) -> bool:
        """
        Feedback 디렉토리에서 reviewed=True 케이스만 로드
        FAISS가 활성화되어 있으면 자동으로 벡터 인덱스 구축
        
        Args:
            force_rebuild: 강제 재구축 여부
            
        Returns: 성공 여부
        """
        if not os.path.exists(self.feedback_dir):
            log.warning(f"Feedback 디렉토리 없음: {self.feedback_dir}")
            return False

        # 모든 *_low_confidence_cases.csv 파일 수집
        files = sorted(glob.glob(os.path.join(self.feedback_dir, "*_low_confidence_cases.csv")))
        if not files:
            log.warning(f"Feedback 디렉토리에 CSV 파일 없음: {self.feedback_dir}")
            return False

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                
                # reviewed=True이고 feedback_label이 있는 케이스만
                if "reviewed" not in df.columns or "feedback_label" not in df.columns:
                    continue
                
                reviewed = df[
                    (df["reviewed"] == True) & 
                    (df["feedback_label"].notna()) & 
                    (df["feedback_label"] != "")
                ].copy()
                
                if len(reviewed) > 0:
                    # 소스 파일 정보 추가
                    reviewed["__fb_source"] = os.path.basename(f)
                    dfs.append(reviewed)
                    log.info(f"로드: {os.path.basename(f)} ({len(reviewed)} reviewed cases)")
                    
            except Exception as e:
                log.warning(f"로드 실패: {f} → {e}")
                continue

        if not dfs:
            log.warning("로드된 reviewed 케이스 없음")
            return False

        self.fb_df = pd.concat(dfs, ignore_index=True)
        
        # 필수 컬럼 확인
        required = ["feedback_label", "case_id"]
        missing = [c for c in required if c not in self.fb_df.columns]
        if missing:
            log.error(f"필수 컬럼 누락: {missing}")
            return False
        
        # 기본값 설정
        if "feedback_reason" not in self.fb_df.columns:
            self.fb_df["feedback_reason"] = ""
        if "feedback_confidence" not in self.fb_df.columns:
            self.fb_df["feedback_confidence"] = 0

        self.is_loaded = True
        log.info(f"✅ Feedback Base 로드 완료: {len(self.fb_df)} reviewed cases")
        
        # FAISS 벡터 인덱스 자동 구축
        if self.use_faiss:
            log.info("\n" + "=" * 70)
            log.info("FAISS 벡터 인덱스 자동 구축 시작")
            log.info("=" * 70)
            
            # 캐시 확인
            data_hash = self._compute_data_hash(self.fb_df)
            cache_paths = self._get_cache_paths(data_hash)
            
            # 캐시 존재 여부 확인
            cache_exists = all(os.path.exists(p) for p in cache_paths.values())
            
            if cache_exists and not force_rebuild and not self.auto_rebuild:
                log.info(f"📦 캐시에서 로드 (hash={data_hash[:8]})")
                try:
                    success = self._load_from_cache(cache_paths)
                    if success:
                        log.info("✅ FAISS 벡터 인덱스 캐시 로드 완료")
                        return True
                except Exception as e:
                    log.warning(f"캐시 로드 실패, 재구축: {e}")
            
            # 자동 재구축 로직
            if self.auto_rebuild and cache_exists:
                cached_hash = self._load_cached_hash(cache_paths)
                if cached_hash != data_hash:
                    log.info(f"🔨 새 피드백 감지 (hash 변경) → 재구축")
                    force_rebuild = True
            
            # FAISS 인덱스 구축
            success = self.build_faiss_index(force_rebuild=force_rebuild)
            if success:
                log.info("✅ FAISS 벡터 인덱스 구축 완료")
            else:
                log.warning("⚠️ FAISS 벡터 인덱스 구축 실패 - 기본 Feedback만 사용")
        
        return True

    def build_faiss_index(self, feature_cols: List[str] = None, 
                         top_k: int = 5, force_rebuild: bool = False) -> bool:
        """
        FAISS 벡터 인덱스 구축 (22개 Feature + SHAP Top-5)
        KB와 동일한 구조
        """
        if not self.is_loaded or self.fb_df is None:
            log.error("Feedback Base가 로드되지 않음")
            return False
        
        if not HAS_FAISS:
            log.warning("FAISS 미설치 - 벡터 인덱스 스킵")
            return False
        
        # Feature 컬럼 결정
        if feature_cols is None:
            feature_cols = NETWORK_SECURITY_FEATURES
        
        available_features = [f for f in feature_cols if f in self.fb_df.columns]
        missing_features = [f for f in feature_cols if f not in self.fb_df.columns]
        
        if missing_features:
            log.warning(f"누락된 Feature ({len(missing_features)}개)")
        
        if not available_features:
            log.error("사용 가능한 Feature 없음")
            return False
        
        log.info(f"📊 사용 Feature: {len(available_features)}/{len(feature_cols)}")
        log.info(f"🎮 GPU 사용: {'ON' if self.use_gpu else 'OFF'}")
        log.info(f"🔧 인덱스 타입: {self.index_type}")
        log.info(f"📦 Feedback 크기: {len(self.fb_df)}개")
        
        start_total = time.time()
        
        # Step 1: Feature 벡터 추출 및 정규화
        log.info(f"  Step 1/5: Feature 정규화...")
        start = time.time()
        
        feature_matrix = self.fb_df[available_features].values.astype(np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            normalized_features = self.scaler.fit_transform(feature_matrix).astype(np.float32)
        else:
            self.scaler = None
            normalized_features = feature_matrix
        
        log.info(f"    ✅ 완료 ({time.time()-start:.2f}초)")
        
        # Step 2: SHAP One-Hot 인코딩 (벡터화)
        log.info(f"  Step 2/5: SHAP One-Hot 인코딩 (벡터화)...")
        start = time.time()
        
        self.shap_feature_vocab = NETWORK_SECURITY_FEATURES
        shap_to_idx = {feat: i for i, feat in enumerate(self.shap_feature_vocab)}
        
        # 빈 행렬 생성
        shap_onehot_matrix = np.zeros((len(self.fb_df), 22), dtype=np.float32)
        
        # 각 SHAP Top-K 컬럼을 벡터화 처리
        for rank in range(1, top_k + 1):
            feat_col = f"shap_top{rank}_feature"
            
            if feat_col not in self.fb_df.columns:
                continue
            
            # 전체 컬럼을 한번에 가져옴
            feat_series = self.fb_df[feat_col].astype(str)
            
            # 유효한 값만 필터링
            valid_mask = ~feat_series.isin(["", "nan", "None", "NaN"])
            
            # 각 Feature에 대해 매칭되는 행 찾기
            for feat_name, feat_idx in shap_to_idx.items():
                match_mask = valid_mask & (feat_series == feat_name)
                matching_rows = self.fb_df.index[match_mask].to_numpy()
                
                # One-Hot 설정
                if len(matching_rows) > 0:
                    shap_onehot_matrix[matching_rows, feat_idx] = 1.0
        
        log.info(f"    ✅ 완료 ({time.time()-start:.2f}초)")
        
        # Step 3: 복합 벡터 생성
        log.info(f"  Step 3/5: 복합 벡터 생성...")
        start = time.time()
        
        composite_vectors = np.hstack([normalized_features, shap_onehot_matrix]).astype(np.float32)
        
        log.info(f"    ├─ Feature 차원: {normalized_features.shape[1]}")
        log.info(f"    ├─ SHAP 차원: {shap_onehot_matrix.shape[1]} (22개 고정)")
        log.info(f"    └─ 총 차원: {composite_vectors.shape[1]}")
        log.info(f"    ✅ 완료 ({time.time()-start:.2f}초)")
        
        # Step 4: 메타데이터 구축 (텍스트 포함)
        log.info(f"  Step 4/5: 메타데이터 구축 (텍스트 포함)...")
        start = time.time()
        
        self.metadata = []
        
        # SHAP 컬럼들
        shap_cols = [f"shap_top{i}_feature" for i in range(1, top_k + 1) 
                    if f"shap_top{i}_feature" in self.fb_df.columns]
        
        for idx in range(len(self.fb_df)):
            row = self.fb_df.iloc[idx]
            
            meta = {
                "idx": idx,
                "case_id": str(row.get("case_id", "")),
                "A_ip": str(row.get("A_ip", "")),
                "B_ip": str(row.get("B_ip", "")),
                
                # ✅ Feedback 고유 필드
                "feedback_label": str(row.get("feedback_label", "")),
                "feedback_reason": str(row.get("feedback_reason", "")),  # 텍스트
                "feedback_confidence": int(row.get("feedback_confidence", 0)),
                "reviewed_by": str(row.get("reviewed_by", "")),
                "review_date": str(row.get("review_date", "")),
                
                # 소스 정보
                "source_file": str(row.get("__fb_source", "")),
                
                # SHAP features
                "shap_features": {}
            }
            
            # SHAP features 추출
            for rank, col in enumerate(shap_cols, 1):
                feat_name = str(row[col])
                if feat_name and feat_name not in ("", "nan", "None"):
                    if feat_name in shap_to_idx:
                        meta["shap_features"][feat_name] = rank
            
            self.metadata.append(meta)
        
        log.info(f"    ✅ 완료 ({time.time()-start:.2f}초)")
        
        # Step 5: FAISS 인덱스 생성
        log.info(f"  Step 5/5: FAISS 인덱스 생성...")
        start = time.time()
        
        dimension = composite_vectors.shape[1]
        self.feature_columns = available_features
        
        self.faiss_index = self._create_faiss_index(dimension)
        
        log.info(f"    ✅ 인덱스 생성 완료 ({time.time()-start:.2f}초)")
        
        # IVF Train
        if self.index_type == "IVF":
            log.info(f"    🔧 IVF 학습 시작 (GPU={'ON' if self.use_gpu else 'OFF'})...")
            start = time.time()
            
            self.faiss_index.train(composite_vectors)
            
            elapsed = time.time() - start
            log.info(f"    ✅ IVF 학습 완료 ({elapsed:.2f}초)")
        
        # 벡터 추가
        log.info(f"    🔧 벡터 추가 시작 ({len(composite_vectors)}개)...")
        start = time.time()
        
        self.faiss_index.add(composite_vectors)
        
        elapsed = time.time() - start
        vec_per_sec = len(composite_vectors) / elapsed if elapsed > 0 else 0
        log.info(f"    ✅ 벡터 추가 완료 ({elapsed:.2f}초, {vec_per_sec:.0f} vec/s)")
        
        self.faiss_built = True
        
        # 캐시 저장
        data_hash = self._compute_data_hash(self.fb_df)
        cache_paths = self._get_cache_paths(data_hash)
        
        try:
            self._save_to_cache(cache_paths, composite_vectors, data_hash)
            log.info(f"    💾 캐시 저장 완료")
        except Exception as e:
            log.warning(f"    ⚠️ 캐시 저장 실패: {e}")
        
        total_elapsed = time.time() - start_total
        log.info(f"\n✅ FAISS 인덱스 구축 완료: {len(self.metadata)}개 케이스 ({total_elapsed:.2f}초)")
        
        return True

    def search_similar_cases(self, 
                            query_features: np.ndarray, 
                            query_shap_features: Dict[str, int],
                            k: int = 10) -> List[Tuple[int, float, Dict]]:
        """
        FAISS로 유사 Feedback 케이스 검색
        
        Args:
            query_features: 쿼리 feature 벡터 (정규화 전)
            query_shap_features: 쿼리 SHAP features {name: rank}
            k: 반환할 후보 수
            
        Returns:
            [(fb_index, distance, metadata), ...]
            metadata에는 feedback_reason, feedback_label 등 포함
        """
        if not self.faiss_built:
            log.warning("FAISS 인덱스 미구축")
            return []
        
        # 쿼리 정규화
        query_features = np.nan_to_num(query_features, nan=0.0).astype(np.float32)
        if self.scaler is not None:
            normalized_query = self.scaler.transform(query_features.reshape(1, -1)).astype(np.float32)
        else:
            normalized_query = query_features.reshape(1, -1).astype(np.float32)
        
        # SHAP One-Hot (22개 고정)
        query_shap_onehot = np.zeros((1, 22), dtype=np.float32)
        shap_to_idx = {feat: i for i, feat in enumerate(self.shap_feature_vocab)}
        
        for feat_name in query_shap_features.keys():
            if feat_name in shap_to_idx:
                feat_idx = shap_to_idx[feat_name]
                query_shap_onehot[0, feat_idx] = 1.0
        
        # 복합 벡터
        query_composite = np.hstack([normalized_query, query_shap_onehot]).astype(np.float32)
        
        # 검색
        if self.index_type == "IVF":
            if hasattr(self.faiss_index, 'nprobe'):
                self.faiss_index.nprobe = min(10, self.n_clusters)
        
        distances, indices = self.faiss_index.search(query_composite, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            if idx >= 0 and idx < len(self.metadata):
                results.append((idx, dist, self.metadata[idx]))
        
        return results

    def generate_llm_context(self, 
                            similar_cases: List[Tuple[int, float, Dict]],
                            max_cases: int = 3) -> str:
        """
        검색된 유사 사례들을 LLM 프롬프트로 변환
        
        Args:
            similar_cases: search_similar_cases() 결과
            max_cases: 최대 포함 케이스 수
            
        Returns:
            LLM 프롬프트 문자열
        """
        if not similar_cases:
            return ""
        
        context_parts = ["## 유사 피드백 사례 분석\n"]
        
        for i, (idx, distance, meta) in enumerate(similar_cases[:max_cases], 1):
            # 유사도 점수 (거리를 0~1 점수로 변환)
            similarity_score = 1.0 / (1.0 + distance)
            
            case_text = f"""
### 사례 {i} (유사도: {similarity_score:.3f})
- **판정**: {meta.get('feedback_label', 'Unknown')}
- **신뢰도**: {meta.get('feedback_confidence', 0)}/5
- **분석 내용**: {meta.get('feedback_reason', '(설명 없음)')}
- **네트워크**: {meta.get('A_ip', 'N/A')} → {meta.get('B_ip', 'N/A')}
- **케이스 ID**: {meta.get('case_id', 'N/A')}
- **검토 날짜**: {meta.get('review_date', 'N/A')}
"""
            context_parts.append(case_text.strip())
        
        return "\n".join(context_parts)

    def get_stats(self) -> Dict[str, any]:
        """Feedback Base 통계 반환"""
        if not self.is_loaded or self.fb_df is None:
            return {}
        
        stats = {
            "total": len(self.fb_df),
            "labels": self.fb_df["feedback_label"].value_counts().to_dict() if "feedback_label" in self.fb_df.columns else {},
            "files_loaded": len(glob.glob(os.path.join(self.feedback_dir, "*_low_confidence_cases.csv"))),
            "faiss_enabled": self.faiss_built,
            "gpu_enabled": self.use_gpu,
        }
        
        if self.faiss_built:
            stats["faiss_index_type"] = self.index_type
            stats["faiss_dimension"] = len(self.feature_columns) + 22
            stats["feature_dim"] = len(self.feature_columns)
            stats["shap_vocab_dim"] = 22
        
        # Confidence 분포
        if "feedback_confidence" in self.fb_df.columns:
            stats["confidence_distribution"] = self.fb_df["feedback_confidence"].value_counts().to_dict()
        
        return stats

    # ========================================
    # FAISS 인덱스 생성 (KB와 동일)
    # ========================================
    
    def _create_faiss_index(self, dimension: int):
        """FAISS 인덱스 생성 (GPU 전송 포함)"""
        # CPU 인덱스 생성
        if self.index_type == "Flat":
            index = faiss.IndexFlatL2(dimension)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatL2(dimension)
            n_clusters = min(self.n_clusters, max(1, len(self.metadata) // 10))
            index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        elif self.index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 40
        else:
            index = faiss.IndexFlatL2(dimension)
        
        # GPU 전송
        if self.use_gpu:
            try:
                if self.gpu_resources is None:
                    self.gpu_resources = faiss.StandardGpuResources()
                    self.gpu_resources.setTempMemory(1024 * 1024 * 1024)  # 1GB
                
                index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                log.info(f"      ✅ GPU로 인덱스 전송 성공 (Device 0)")
                
            except Exception as e:
                log.error(f"      ❌ GPU 전송 실패: {e}")
                log.error(f"      ⚠️ CPU 모드로 fallback합니다")
                self.use_gpu = False
        
        return index

    # ========================================
    # 캐시 관련 메서드
    # ========================================
    
    def _compute_data_hash(self, fb_df: pd.DataFrame) -> str:
        """데이터 해시 계산"""
        key_parts = [
            f"rows:{len(fb_df)}",
            f"cols:{len(self.feature_columns) if self.feature_columns else 0}",
        ]
        
        # case_id 기반 해시 (reviewed 케이스 추적)
        if "case_id" in fb_df.columns:
            case_ids = sorted(fb_df["case_id"].astype(str).unique())
            case_hash = hashlib.md5(",".join(case_ids).encode()).hexdigest()[:16]
            key_parts.append(f"cases:{case_hash}")
        
        # review_date 기반 해시 (최신성 추적)
        if "review_date" in fb_df.columns:
            dates = fb_df["review_date"].astype(str).unique()
            date_hash = hashlib.md5(",".join(sorted(dates)).encode()).hexdigest()[:8]
            key_parts.append(f"dates:{date_hash}")
        
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def _get_cache_paths(self, data_hash: str) -> Dict[str, str]:
        """캐시 파일 경로"""
        prefix = os.path.join(self.cache_dir, f"feedback_{data_hash}")
        return {
            "index": f"{prefix}_index.faiss",
            "scaler": f"{prefix}_scaler.pkl",
            "metadata": f"{prefix}_metadata.pkl",
            "config": f"{prefix}_config.pkl",
            "hash": f"{prefix}_hash.txt",
        }
    
    def _save_to_cache(self, cache_paths: Dict[str, str], vectors: np.ndarray, data_hash: str):
        """캐시 저장"""
        # GPU 인덱스는 CPU로 변환 후 저장
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            faiss.write_index(cpu_index, cache_paths["index"])
        else:
            faiss.write_index(self.faiss_index, cache_paths["index"])
        
        if self.scaler is not None:
            with open(cache_paths["scaler"], "wb") as f:
                pickle.dump(self.scaler, f)
        
        with open(cache_paths["metadata"], "wb") as f:
            pickle.dump(self.metadata, f)
        
        config = {
            "feature_columns": self.feature_columns,
            "shap_feature_vocab": self.shap_feature_vocab,
            "index_type": self.index_type,
            "n_clusters": self.n_clusters,
        }
        with open(cache_paths["config"], "wb") as f:
            pickle.dump(config, f)
        
        # 해시 저장
        with open(cache_paths["hash"], "w") as f:
            f.write(data_hash)
    
    def _load_from_cache(self, cache_paths: Dict[str, str]) -> bool:
        """캐시 로드"""
        with open(cache_paths["config"], "rb") as f:
            config = pickle.load(f)
        
        self.feature_columns = config["feature_columns"]
        self.shap_feature_vocab = config.get("shap_feature_vocab", NETWORK_SECURITY_FEATURES)
        
        self.faiss_index = faiss.read_index(cache_paths["index"])
        
        # GPU로 전송
        if self.use_gpu:
            try:
                if self.gpu_resources is None:
                    self.gpu_resources = faiss.StandardGpuResources()
                    self.gpu_resources.setTempMemory(1024 * 1024 * 1024)
                
                self.faiss_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.faiss_index)
                log.info("✅ 캐시된 인덱스를 GPU로 전송 완료")
            except Exception as e:
                log.warning(f"⚠️ GPU 전송 실패, CPU로 사용: {e}")
                self.use_gpu = False
        
        if os.path.exists(cache_paths["scaler"]):
            with open(cache_paths["scaler"], "rb") as f:
                self.scaler = pickle.load(f)
        
        with open(cache_paths["metadata"], "rb") as f:
            self.metadata = pickle.load(f)
        
        self.faiss_built = True
        return True
    
    def _load_cached_hash(self, cache_paths: Dict[str, str]) -> Optional[str]:
        """캐시된 해시 로드"""
        hash_file = cache_paths.get("hash")
        if hash_file and os.path.exists(hash_file):
            with open(hash_file, "r") as f:
                return f.read().strip()
        return None

    def __len__(self) -> int:
        return len(self.fb_df) if self.is_loaded and self.fb_df is not None else 0

    def __repr__(self) -> str:
        if not self.is_loaded:
            return "FeedbackBase(not loaded)"
        stats = self.get_stats()
        gpu_info = f", GPU={'ON' if self.use_gpu else 'OFF'}" if self.faiss_built else ""
        faiss_info = f", FAISS({self.index_type}){gpu_info}" if self.faiss_built else ""
        return f"FeedbackBase(total={stats.get('total', 0)}, labels={stats.get('labels', {})}{faiss_info})"