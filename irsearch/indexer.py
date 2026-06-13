"""ES 색인 — 코퍼스(.txt) → nori 분석기 인덱스에 bulk 색인.

인덱스 매핑:
- text/section/doc_title : nori 형태소 분석(한자→한글 reading, 복합어 분해, 조사 제거)
- product_type/insurer/doc_id : keyword(필터·집계용)
"""
from __future__ import annotations
import json

from elasticsearch import Elasticsearch, helpers

from .config import (CORPUS_DIR, INDEX_DIR, STATS_PATH, CHUNK_MAX_CHARS,
                     ES_URL, ES_INDEX, EMBED_DIM, EMBED_MODEL)
from .chunking import parse_doc
from .embedder import encode

SETTINGS = {
    "analysis": {
        "tokenizer": {
            "nori_mixed": {
                "type": "nori_tokenizer",
                "decompound_mode": "mixed",
                # 사용자 사전: nori가 모르는 신조어를 한 단어로 등재
                # → '코로나도'가 '코로나'+'도(조사)'로 분리되고 조사는 POS 필터가 제거
                # '백만종', 정식 상품명은 고유명사 → 통째로 한 토큰(형분 방지)
                "user_dictionary_rules": ["코로나", "백만종", "백만인을위한종신보험"],
            }
        },
        "filter": {
            # 유의어(검색 시점 확장): '코로나'→'감염병', '백만종'→정식 상품명(고유명사 통째)
            "ko_synonym": {
                "type": "synonym_graph",
                "synonyms": ["코로나, 감염병", "백만종, 백만인을위한종신보험"],
            }
        },
        "analyzer": {
            "korean": {                       # 색인용 — 유의어 미적용
                "type": "custom",
                "tokenizer": "nori_mixed",
                "filter": ["nori_readingform", "lowercase", "nori_part_of_speech"],
            },
            "korean_search": {                # 검색용 — 유의어 적용
                "type": "custom",
                "tokenizer": "nori_mixed",
                "filter": ["nori_readingform", "lowercase",
                           "nori_part_of_speech", "ko_synonym"],
            },
        },
    }
}

# 색인은 korean, 검색은 korean_search(유의어 적용)
_TEXT = {"type": "text", "analyzer": "korean", "search_analyzer": "korean_search"}
MAPPINGS = {
    "properties": {
        "doc_id": {"type": "keyword"},
        "doc_title": {**_TEXT, "fields": {"raw": {"type": "keyword"}}},
        "gwan": _TEXT,
        "section": _TEXT,
        "product_type": {"type": "keyword"},
        "insurer": {"type": "keyword"},
        "chunk_id": {"type": "keyword"},
        "text": _TEXT,
        # 시맨틱 검색용 임베딩(HNSW 근사 kNN, 코사인 유사도)
        "vector": {
            "type": "dense_vector",
            "dims": EMBED_DIM,
            "index": True,
            "similarity": "cosine",
        },
    }
}


def client() -> Elasticsearch:
    return Elasticsearch(ES_URL, request_timeout=30)


def build_index() -> dict:
    es = client()
    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"코퍼스가 비어 있습니다: {CORPUS_DIR}")

    if es.indices.exists(index=ES_INDEX):
        es.indices.delete(index=ES_INDEX)
    es.indices.create(index=ES_INDEX, settings=SETTINGS, mappings=MAPPINGS)

    # 1) 청크 파싱
    chunks = []
    for f in files:
        chunks.extend(parse_doc(f, doc_id=f.stem, max_chars=CHUNK_MAX_CHARS))

    # 2) 임베딩(상품명+조제목+본문을 합쳐 문맥 보강)
    texts = [f"{c.doc_title} {c.section}\n{c.text}" for c in chunks]
    print(f"[embed] {len(texts)}개 청크 임베딩 중... (model={EMBED_MODEL})")
    vectors = encode(texts, show_progress=True)

    # 3) bulk 색인(BM25 필드 + 벡터 동시)
    def actions():
        for c, v in zip(chunks, vectors):
            yield {
                "_index": ES_INDEX, "_id": c.chunk_id,
                "_source": {
                    "doc_id": c.doc_id, "doc_title": c.doc_title,
                    "gwan": c.gwan, "section": c.section,
                    "product_type": c.product_type,
                    "insurer": c.insurer, "chunk_id": c.chunk_id, "text": c.text,
                    "vector": v,
                },
            }

    helpers.bulk(es, actions())
    es.indices.refresh(index=ES_INDEX)

    n_chunks = es.count(index=ES_INDEX)["count"]
    stats = {"documents": len(files), "chunks": n_chunks, "index": ES_INDEX,
             "embed_model": EMBED_MODEL, "embed_dim": EMBED_DIM}
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(build_index())
