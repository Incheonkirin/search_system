"""크로스 인코더 리랭커 — BAAI/bge-reranker-v2-m3 (로컬).

bi-encoder(임베딩+코사인)와 달리 질의와 문서를 *함께* 한 트랜스포머에 넣어
관련도 점수를 낸다. 색인/kNN은 불가능 → 1차 검색이 추린 후보에만 적용.
최초 호출 시 1회 로드(싱글톤), Apple Silicon이면 MPS.
"""
from __future__ import annotations
import threading

_model = None
_lock = threading.Lock()

MODEL = "BAAI/bge-reranker-v2-m3"


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
                from sentence_transformers import CrossEncoder
                _model = CrossEncoder(MODEL, max_length=512, device=_device())
    return _model


def score(query: str, passages: list[str]) -> list[float]:
    """(query, passage) 쌍들의 관련도 점수 리스트."""
    if not passages:
        return []
    model = get_model()
    preds = model.predict([(query, p) for p in passages])
    return [float(x) for x in preds]
