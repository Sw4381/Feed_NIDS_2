# -*- coding: utf-8 -*-
"""
Knowledge Base Manager with FAISS Vector Store Integration
Train 시점의 labeled 사례들을 로드하고 FAISS 벡터 인덱스로 관리하는 모듈
"""
from __future__ import annotations
import os
import glob
import pickle
import hashlib
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from tools.base import get_logger

log = get_logger("KnowledgeBase")

# 네트워크 보안 Feature 22개 (고정)
NETWORK_SECURITY_FEATURES = [
    'alpahbet_cnt_payload_sum',      # Payload 알파벳 문자 수
    'alpha_cnt_dns_query_sum',       # DNS 쿼리 알파벳 문자 수
    'client_extensions_cnt',          # 클라이언트 확장 수
    'entropys_avg',                   # 엔트로피 평균
    'flow_delta_times_sum',          # Flow 시간 델타 합
    'flow_duration_seconds',         # Flow 지속 시간 (초)
    'flow_stdev_time',               # Flow 시간 표준편차
    'nonascii_cnt_dns_query_sum',    # DNS 쿼리 비ASCII 문자 수
    'nonascii_cnt_payload_sum',      # Payload 비ASCII 문자 수
    'number_cnt_dns_query_sum',      # DNS 쿼리 숫자 문자 수
    'number_cnt_payload_sum',        # Payload 숫자 문자 수
    'payload_len_max',               # Payload 최대 길이
    'payload_len_min',               # Payload 최소 길이
    'payload_lens_sum',              # Payload 길이 합
    'payload_packets_cnt',           # Payload 패킷 수
    'query_response_ttls_sum',       # DNS TTL 합
    'server_certificates_cnt',        # 서버 인증서 수
    'server_extensions_cnt',          # 서버 확장 수
    'special_cnt_dns_query_sum',     # DNS 쿼리 특수문자 수
    'special_cnt_payload_sum',       # Payload 특수문자 수
    'tls_SAN_cnt',                   # TLS SAN 수
    'total_packets_cnt'              # 총 패킷 수
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


class KnowledgeBase:
    """
    Train_Cases 디렉토리에서 labeled 사례를 로드하고 FAISS 벡터 인덱스로 관리
    
    Features:
    - 기존 KB 로드 및 관리 (backward compatible)
    - FAISS 벡터 인덱스 자동 구축 (22개 Feature + SHAP Top-5)
    - 캐시 시스템 (hash 기반)
    - GPU 지원 (선택 사항)
    """
    
    def __init__(self, 
                 train_cases_dir: str = "/Train_Cases",
                 use_faiss: bool = True,
                 cache_dir: str = "./cache",
                 index_type: str = "IVF",
                 n_clusters: int = 100,
                 use_gpu: bool = False):
        """
        Args:
            train_cases_dir: Train Cases 디렉토리 경로
            use_faiss: FAISS 벡터 인덱스 사용 여부
            cache_dir: 캐시 디렉토리
            index_type: FAISS 인덱스 타입 ("Flat", "IVF", "HNSW")
            n_clusters: IVF 클러스터 수
            use_gpu: GPU 사용 여부
        """
        self.train_cases_dir = train_cases_dir
        self.kb_df = None
        self.is_loaded = False
        
        # FAISS 설정
        self.use_faiss = use_faiss and HAS_FAISS
        self.cache_dir = cache_dir
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu and HAS_FAISS
        
        # FAISS 벡터 저장소
        self.faiss_index = None
        self.scaler = None
        self.feature_columns = []
        self.shap_feature_vocab = []
        self.metadata = []
        self.faiss_built = False
        
        if self.use_faiss:
            os.makedirs(cache_dir, exist_ok=True)

    def load(self) -> bool:
        """
        Train_Cases 디렉토리에서 모든 CSV 파일 로드
        FAISS가 활성화되어 있으면 자동으로 벡터 인덱스 구축
        
        Returns: 성공 여부
        """
        if not os.path.exists(self.train_cases_dir):
            log.warning(f"Train_Cases 디렉토리 없음: {self.train_cases_dir}")
            return False

        files = sorted(glob.glob(os.path.join(self.train_cases_dir, "*.csv")))
        if not files:
            log.warning(f"Train_Cases에 CSV 파일 없음: {self.train_cases_dir}")
            return False

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f, low_memory=False)
                if "label" in df.columns and len(df) > 0:
                    dfs.append(df)
                    log.info(f"로드: {os.path.basename(f)} ({len(df)} rows)")
            except Exception as e:
                log.warning(f"로드 실패: {f} → {e}")
                continue

        if not dfs:
            log.warning("로드된 labeled 사례 없음")
            return False

        self.kb_df = pd.concat(dfs, ignore_index=True)
        
        # 필수 컬럼 확인
        required = ["label"]
        missing = [c for c in required if c not in self.kb_df.columns]
        if missing:
            log.error(f"필수 컬럼 누락: {missing}")
            return False

        self.is_loaded = True
        log.info(f"✅ Knowledge Base 로드 완료: {len(self.kb_df)} rows")
        
        # FAISS 벡터 인덱스 자동 구축
        if self.use_faiss:
            log.info("\n" + "=" * 60)
            log.info("FAISS 벡터 인덱스 자동 구축 시작")
            log.info("=" * 60)
            success = self.build_faiss_index()
            if success:
                log.info("✅ FAISS 벡터 인덱스 구축 완료")
            else:
                log.warning("⚠️ FAISS 벡터 인덱스 구축 실패 - 기본 KB만 사용")
        
        return True

    def build_faiss_index(self, feature_cols: List[str] = None, 
                          top_k: int = 5, force_rebuild: bool = False) -> bool:
        """
        FAISS 벡터 인덱스 구축 (22개 Feature + SHAP Top-5)
        
        Args:
            feature_cols: Feature 컬럼 (None이면 기본 22개 사용)
            top_k: SHAP Top-K
            force_rebuild: 강제 재구축
            
        Returns:
            성공 여부
        """
        if not self.is_loaded or self.kb_df is None:
            log.error("KB가 로드되지 않음")
            return False
        
        if not HAS_FAISS:
            log.warning("FAISS 미설치 - 벡터 인덱스 스킵")
            return False
        
        # Feature 컬럼 결정
        if feature_cols is None:
            feature_cols = NETWORK_SECURITY_FEATURES
        
        # KB에 실제 존재하는 Feature만 필터링
        available_features = [f for f in feature_cols if f in self.kb_df.columns]
        missing_features = [f for f in feature_cols if f not in self.kb_df.columns]
        
        if missing_features:
            log.warning(f"누락된 Feature ({len(missing_features)}개)")
        
        if not available_features:
            log.error("사용 가능한 Feature 없음")
            return False
        
        log.info(f"사용 Feature: {len(available_features)}/{len(feature_cols)}")
        
        # 캐시 확인
        data_hash = self._compute_data_hash(self.kb_df, available_features)
        cache_paths = self._get_cache_paths(data_hash)
        
        if not force_rebuild and all(os.path.exists(p) for p in cache_paths.values()):
            log.info(f"📦 캐시에서 로드 (hash={data_hash[:8]})")
            try:
                return self._load_from_cache(cache_paths)
            except Exception as e:
                log.warning(f"캐시 로드 실패, 재구축: {e}")
        
        log.info(f"🔨 벡터 인덱스 구축 중...")
        
        # Step 1: Feature 벡터 추출 및 정규화
        feature_matrix = self.kb_df[available_features].values.astype(np.float32)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            normalized_features = self.scaler.fit_transform(feature_matrix).astype(np.float32)
        else:
            self.scaler = None
            normalized_features = feature_matrix
        
        # Step 2: SHAP Top-K → One-Hot 인코딩
        all_shap_features = set()
        shap_feature_lists = []
        
        for idx, row in self.kb_df.iterrows():
            row_shap_features = []
            for i in range(1, top_k + 1):
                feat_col = f"shap_top{i}_feature"
                if feat_col in row:
                    feat_name = str(row[feat_col])
                    if feat_name and feat_name not in ("", "nan", "None"):
                        row_shap_features.append(feat_name)
                        all_shap_features.add(feat_name)
            shap_feature_lists.append(row_shap_features)
        
        self.shap_feature_vocab = sorted(all_shap_features)
        shap_to_idx = {feat: i for i, feat in enumerate(self.shap_feature_vocab)}
        
        shap_onehot_matrix = np.zeros((len(self.kb_df), len(self.shap_feature_vocab)), dtype=np.float32)
        for row_idx, shap_feats in enumerate(shap_feature_lists):
            for feat_name in shap_feats:
                feat_idx = shap_to_idx[feat_name]
                shap_onehot_matrix[row_idx, feat_idx] = 1.0
        
        # Step 3: 복합 벡터 생성
        composite_vectors = np.hstack([normalized_features, shap_onehot_matrix]).astype(np.float32)
        
        log.info(f"  ├─ Feature 차원: {normalized_features.shape[1]}")
        log.info(f"  ├─ SHAP Vocab 차원: {shap_onehot_matrix.shape[1]}")
        log.info(f"  └─ 총 차원: {composite_vectors.shape[1]}")
        
        # Step 4: 메타데이터 저장
        self.metadata = []
        for idx, row in self.kb_df.iterrows():
            meta = {
                "idx": idx,
                "A_ip": str(row.get("A_ip", "")),
                "B_ip": str(row.get("B_ip", "")),
                "label": str(row.get("label", "")),
                "case_id": str(row.get("case_id", "")),
                "shap_features": {}
            }
            for i in range(1, top_k + 1):
                feat_col = f"shap_top{i}_feature"
                if feat_col in row:
                    feat_name = str(row[feat_col])
                    if feat_name and feat_name not in ("", "nan", "None"):
                        meta["shap_features"][feat_name] = i
            self.metadata.append(meta)
        
        # Step 5: FAISS 인덱스 생성
        dimension = composite_vectors.shape[1]
        self.feature_columns = available_features
        
        self.faiss_index = self._create_faiss_index(dimension)
        
        if self.index_type == "IVF":
            self.faiss_index.train(composite_vectors)
        
        self.faiss_index.add(composite_vectors)
        self.faiss_built = True
        
        # 캐시 저장
        try:
            self._save_to_cache(cache_paths, composite_vectors)
        except Exception as e:
            log.warning(f"캐시 저장 실패: {e}")
        
        return True

    def _create_faiss_index(self, dimension: int):
        """FAISS 인덱스 생성"""
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
        
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception as e:
                log.warning(f"GPU 변환 실패: {e}")
        
        return index

    def search_similar_cases(self, query_features: np.ndarray, 
                            query_shap_features: Dict[str, int],
                            k: int = 100) -> List[Tuple[int, float]]:
        """
        FAISS로 유사 케이스 검색
        
        Args:
            query_features: 쿼리 feature 벡터 (정규화 전)
            query_shap_features: 쿼리 SHAP features {name: rank}
            k: 반환할 후보 수
            
        Returns:
            [(kb_index, distance), ...]
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
        
        # SHAP One-Hot
        query_shap_onehot = np.zeros((1, len(self.shap_feature_vocab)), dtype=np.float32)
        for feat_name in query_shap_features.keys():
            if feat_name in self.shap_feature_vocab:
                feat_idx = self.shap_feature_vocab.index(feat_name)
                query_shap_onehot[0, feat_idx] = 1.0
        
        # 복합 벡터
        query_composite = np.hstack([normalized_query, query_shap_onehot]).astype(np.float32)
        
        # 검색
        if self.index_type == "IVF":
            self.faiss_index.nprobe = min(10, self.n_clusters)
        
        distances, indices = self.faiss_index.search(query_composite, k)
        
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            dist = float(distances[0][i])
            if idx >= 0:
                results.append((idx, dist))
        
        return results

    def get_labeled_cases(self, labels: List[str] = None) -> pd.DataFrame:
        """
        특정 라벨의 사례만 반환 (backward compatible)
        """
        if not self.is_loaded or self.kb_df is None:
            return pd.DataFrame()
        if labels is None:
            return self.kb_df.copy()
        return self.kb_df[self.kb_df["label"].isin(labels)].copy()

    def get_stats(self) -> Dict[str, any]:
        """Knowledge Base 통계 반환"""
        if not self.is_loaded or self.kb_df is None:
            return {}
        
        stats = {
            "total": len(self.kb_df),
            "labels": self.kb_df["label"].value_counts().to_dict() if "label" in self.kb_df.columns else {},
            "files_loaded": len(glob.glob(os.path.join(self.train_cases_dir, "*.csv"))),
            "faiss_enabled": self.faiss_built,
        }
        
        if self.faiss_built:
            stats["faiss_index_type"] = self.index_type
            stats["faiss_dimension"] = len(self.feature_columns) + len(self.shap_feature_vocab)
            stats["feature_dim"] = len(self.feature_columns)
            stats["shap_vocab_dim"] = len(self.shap_feature_vocab)
        
        return stats

    def export_as_feedback_corpus(self, out_dir: str = "./feedback_cases") -> str:
        """
        Knowledge Base를 피드백 코퍼스 형식으로 Export (backward compatible)
        """
        if not self.is_loaded or self.kb_df is None:
            log.error("Knowledge Base 미로드")
            return ""

        os.makedirs(out_dir, exist_ok=True)
        export_df = self.kb_df.copy()
        
        for col, default in [
            ("case_id", ""),
            ("feedback_label", ""),
            ("feedback_confidence", ""),
            ("feedback_reason", ""),
            ("reviewed", True),
            ("needs_review", False),
            ("review_date", ""),
        ]:
            if col not in export_df.columns:
                if col == "reviewed":
                    export_df[col] = True
                elif col == "case_id" and "case_id" not in export_df.columns:
                    export_df[col] = [f"KB_{i:06d}" for i in range(len(export_df))]
                elif col == "feedback_label":
                    export_df[col] = export_df.get("label", "")
                elif col == "feedback_confidence":
                    export_df[col] = 5
                elif col == "feedback_reason":
                    export_df[col] = "(Train Knowledge Base)"
                else:
                    export_df[col] = default

        out_path = os.path.join(out_dir, "Knowledge_Base_Corpus.csv")
        export_df.to_csv(out_path, index=False, encoding="utf-8")
        log.info(f"Knowledge Base Export: {out_path} ({len(export_df)} rows)")
        return out_path

    # ========================================
    # 캐시 관련 메서드
    # ========================================
    
    def _compute_data_hash(self, kb_df: pd.DataFrame, feature_cols: List[str]) -> str:
        """데이터 해시 계산"""
        key_parts = [
            f"rows:{len(kb_df)}",
            f"cols:{len(feature_cols)}",
            f"features:{','.join(sorted(feature_cols))}",
        ]
        if len(kb_df) > 0:
            sample_size = min(100, len(kb_df))
            sample_idx = np.linspace(0, len(kb_df)-1, sample_size, dtype=int)
            sample_data = kb_df.iloc[sample_idx][feature_cols].values
            data_hash = hashlib.md5(sample_data.tobytes()).hexdigest()[:16]
            key_parts.append(f"data:{data_hash}")
        return hashlib.md5("_".join(key_parts).encode()).hexdigest()
    
    def _get_cache_paths(self, data_hash: str) -> Dict[str, str]:
        """캐시 파일 경로"""
        prefix = os.path.join(self.cache_dir, f"kb_{data_hash}")
        return {
            "index": f"{prefix}_index.faiss",
            "scaler": f"{prefix}_scaler.pkl",
            "metadata": f"{prefix}_metadata.pkl",
            "config": f"{prefix}_config.pkl",
        }
    
    def _save_to_cache(self, cache_paths: Dict[str, str], vectors: np.ndarray):
        """캐시 저장"""
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
    
    def _load_from_cache(self, cache_paths: Dict[str, str]) -> bool:
        """캐시 로드"""
        with open(cache_paths["config"], "rb") as f:
            config = pickle.load(f)
        
        self.feature_columns = config["feature_columns"]
        self.shap_feature_vocab = config.get("shap_feature_vocab", [])
        
        self.faiss_index = faiss.read_index(cache_paths["index"])
        
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            except:
                pass
        
        if os.path.exists(cache_paths["scaler"]):
            with open(cache_paths["scaler"], "rb") as f:
                self.scaler = pickle.load(f)
        
        with open(cache_paths["metadata"], "rb") as f:
            self.metadata = pickle.load(f)
        
        self.faiss_built = True
        return True

    def __len__(self) -> int:
        return len(self.kb_df) if self.is_loaded and self.kb_df is not None else 0

    def __repr__(self) -> str:
        if not self.is_loaded:
            return "KnowledgeBase(not loaded)"
        stats = self.get_stats()
        faiss_info = f", FAISS({self.index_type})" if self.faiss_built else ""
        return f"KnowledgeBase(total={stats.get('total', 0)}, labels={stats.get('labels', {})}{faiss_info})"