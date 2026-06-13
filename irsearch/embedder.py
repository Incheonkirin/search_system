"""임베딩 — BGE-M3(로컬, sentence-transformers).

- 모델은 최초 호출 시 1회 로드(싱글톤). Apple Silicon이면 MPS 가속.
- 임베딩은 L2 정규화 → 코사인 유사도 = 내적(ES dense_vector similarity=cosine과 일치).
- bge-m3는 query/passage 프리픽스가 불필요(같은 인코더 사용).
"""
from __future__ import annotations
import threading

from .config import EMBED_MODEL, EMBED_BATCH

_model = None
_lock = threading.Lock()


def _device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer(EMBED_MODEL, device=_device())
                # 긴 약관 조문을 통째 인코딩하면(기본 8192) 매우 느림.
                # 검색 품질 대비 속도를 위해 512 토큰으로 제한(대부분 조는 그 안에 핵심 포함).
                _model.max_seq_length = 512
    return _model


def encode(texts: list[str], batch_size: int = EMBED_BATCH,
           show_progress: bool = False) -> list[list[float]]:
    """문서/문장 리스트 → 정규화 임베딩 리스트."""
    model = get_model()
    vecs = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=show_progress, convert_to_numpy=True)
    return vecs.tolist()


def encode_one(text: str) -> list[float]:
    """질의 한 건 → 정규화 임베딩."""
    return encode([text])[0]
