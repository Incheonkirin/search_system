"""검색 — 어휘(BM25/nori) × 의미(BGE-M3 kNN) 하이브리드.

mode:
- lexical  : BM25 multi_match(nori) 단독
- semantic : 임베딩 kNN(코사인) 단독
- hybrid   : 두 결과를 RRF(Reciprocal Rank Fusion)로 융합 (기본)

RRF: 점수 스케일이 다른 두 랭킹을 '순위'로 합친다.  score = Σ 1/(K + rank)
필드 가중/검색식은 추후 튜닝(요구사항: "검색식은 나중에 조정").
"""
from __future__ import annotations
import re

from elasticsearch import Elasticsearch

from .config import ES_URL, ES_INDEX, RRF_K
from .embedder import encode_one


def _dedup_key(d: dict) -> str:
    """근접중복 판별 키 = '조 제목'(괄호 안)만, 공백/대소문자 무시.
    표준조항은 상품마다 조 번호가 달라도 제목이 같으면 동일 조항이므로,
    조 번호·관·상품을 무시하고 제목으로만 묶는다. (예: 제30조/제47조(위법계약의 해지) → 1개)"""
    sec = d.get("section", "")
    m = re.search(r"[(（【](.+?)[)）】]", sec)
    title = m.group(1) if m else sec
    return re.sub(r"\s+", "", title).lower()

# section(조 제목) 과대부스팅 교정: 2.5→1.5 (스윕 결과 IR 21→25/30 피크).
# 2.5에선 '보험금→보험/금' 같은 분해 조각이 제목에 든 엉뚱한 조가 장악했음.
_BM25_FIELDS = ["text", "section^1.5", "gwan^1.5", "doc_title^1.2"]
_HL = {"fields": {"text": {"fragment_size": 160, "number_of_fragments": 1}}}
_RETRIEVE_N = 30   # 1차 IR(BM25) 후보 수 → 2차 시맨틱 재랭크 대상


class Searcher:
    def __init__(self):
        self.es: Elasticsearch | None = None

    def load(self) -> "Searcher":
        es = Elasticsearch(ES_URL, request_timeout=30)
        if not es.ping():
            raise RuntimeError(f"ES 연결 실패: {ES_URL}")
        self.es = es
        return self

    # ---------------- 개별 검색기 ----------------
    def _bm25(self, query: str, size: int, with_vector: bool = False) -> list[dict]:
        # best_fields + cross_fields 합산(bool should):
        #  - best_fields : 조 제목이 정확히 맞는 경우를 잡음(예: '지급하지 않는 사유')
        #  - cross_fields: 변별어가 본문에 흩어진 경우를 잡음(예: '수술','무효')
        # 둘 중 하나만 쓰면 서로의 케이스를 놓침 → 합산해야 IR recall@30 = 정답있는 29건 100%.
        resp = self.es.search(
            index=ES_INDEX, size=size,
            query={"bool": {"should": [
                {"multi_match": {"query": query, "fields": _BM25_FIELDS, "type": "best_fields"}},
                {"multi_match": {"query": query, "fields": _BM25_FIELDS, "type": "cross_fields"}},
            ]}},
            highlight=_HL,
            source_excludes=None if with_vector else ["vector"])
        return resp["hits"]["hits"]

    def _bm25_terms(self, terms: list[str], size: int, with_vector: bool = False) -> list[dict]:
        """외부 분석기(kiwi/okt 등)가 뽑은 토큰으로 BM25 — whitespace 분석기로 질의해
        nori가 토큰을 재분절하지 못하게 한다(분석기별 분절 차이를 그대로 반영). 형분기 비교용."""
        q = " ".join(t for t in terms if t)
        resp = self.es.search(
            index=ES_INDEX, size=size,
            query={"bool": {"should": [
                {"multi_match": {"query": q, "fields": _BM25_FIELDS,
                                 "type": "best_fields", "analyzer": "whitespace"}},
                {"multi_match": {"query": q, "fields": _BM25_FIELDS,
                                 "type": "cross_fields", "analyzer": "whitespace"}},
            ]}},
            highlight=_HL,
            source_excludes=None if with_vector else ["vector"])
        return resp["hits"]["hits"]

    def _knn(self, query: str, size: int) -> list[dict]:
        qv = encode_one(query)
        resp = self.es.search(
            index=ES_INDEX, size=size,
            knn={"field": "vector", "query_vector": qv,
                 "k": size, "num_candidates": max(size * 4, 100)},
            source_excludes=["vector"])
        return resp["hits"]["hits"]

    # ---------------- 결과 변환 ----------------
    def _candidates(self, query: str, n: int, lex_terms: list[str] | None = None) -> dict:
        """1차 후보 풀(공통) = BM25 top-n ∪ 벡터 top-n. chunk_id로 dedup.
        각 후보에 bm25_rank / vec_rank 기록. 대표 hit은 하이라이트 있는 BM25 우선.
        lex_terms 주어지면 BM25를 그 토큰으로(형분기 비교), 벡터는 항상 원문 임베딩(분석기 무관)."""
        bm25_hits = self._bm25_terms(lex_terms, n) if lex_terms else self._bm25(query, n)
        pool: dict[str, dict] = {}
        for rank, h in enumerate(bm25_hits, 1):
            pool.setdefault(h["_id"], {"hit": h})["bm25_rank"] = rank
        for rank, h in enumerate(self._knn(query, n), 1):
            e = pool.setdefault(h["_id"], {"hit": h})
            e["vec_rank"] = rank
            if "highlight" in e["hit"]:      # BM25 hit(하이라이트 보유)을 대표로
                pass
            elif "highlight" in h:
                e["hit"] = h
        return pool

    @staticmethod
    def _base(h: dict) -> dict:
        s = h["_source"]
        hl = h.get("highlight", {}).get("text", [])
        return {
            "doc_id": s["doc_id"], "doc_title": s["doc_title"],
            "gwan": s.get("gwan", ""), "section": s["section"],
            "section_path": " > ".join(
                p for p in [s["doc_title"], s.get("gwan", ""), s["section"]] if p),
            "product_type": s["product_type"], "insurer": s["insurer"],
            "chunk_id": s["chunk_id"],
            "snippet": (hl[0] if hl else s["text"][:240]),
            "text": s["text"],
        }

    # ---------------- 공개 검색 ----------------
    def search(self, query: str, k: int = 10, mode: str = "rerank",
               collapse: bool = False, lex_terms: list[str] | None = None) -> list[dict]:
        """lex_terms 주어지면 BM25 어휘검색을 그 토큰으로 수행(형분기 비교 Level1).
        벡터/임베딩은 항상 원문 query 기준(분석기 무관)."""
        if self.es is None:
            self.load()

        if mode == "lexical":
            sz = max(k * 6, 60) if collapse else k
            hits = self._bm25_terms(lex_terms, sz) if lex_terms else self._bm25(query, sz)
            rows = [{"score": round(float(h["_score"]), 4),
                     "retrievers": ["bm25"], "bm25_rank": i, "vec_rank": None,
                     **self._base(h)} for i, h in enumerate(hits, 1)]
            return self._finalize(rows, k, collapse)

        if mode == "semantic":
            hits = self._knn(query, max(k * 6, 60) if collapse else k)
            rows = [{"score": round(float(h["_score"]), 4),
                     "retrievers": ["vector"], "bm25_rank": None, "vec_rank": i,
                     **self._base(h)} for i, h in enumerate(hits, 1)]
            return self._finalize(rows, k, collapse)

        if mode == "rerank":
            # 코사인 재랭크: BM25 후보 N개를 질의 임베딩 코사인으로 재정렬
            # (참고용 — 전체 시맨틱과 사실상 동일)
            hits = (self._bm25_terms(lex_terms, _RETRIEVE_N, with_vector=True)
                    if lex_terms else self._bm25(query, _RETRIEVE_N, with_vector=True))
            qv = encode_one(query)
            rows = []
            for rank, h in enumerate(hits, 1):
                v = h["_source"].get("vector")
                cos = sum(a * b for a, b in zip(qv, v)) if v else 0.0
                rows.append((cos, {"score": round(cos, 4), "retrievers": ["bm25→cos"],
                                   "bm25_rank": rank, "vec_rank": None, **self._base(h)}))
            rows.sort(key=lambda x: x[0], reverse=True)
            return self._finalize([d for _, d in rows], k, collapse)

        # 공통 1차 후보 풀 (BM25 ∪ 벡터)
        pool = self._candidates(query, _RETRIEVE_N, lex_terms)

        if mode == "cross":
            # (나) 크로스 인코더: (질의, 조제목+본문) 공동 인코딩으로 후보 재랭크
            from . import reranker
            items = list(pool.values())
            passages = [f"{self._base(e['hit'])['section']} {self._base(e['hit'])['text']}"
                        for e in items]
            scores = reranker.score(query, passages)
            rows = []
            for e, sc in zip(items, scores):
                rows.append((sc, {"score": round(float(sc), 4),
                                  "retrievers": ["bm25∪vec→cross"],
                                  "bm25_rank": e.get("bm25_rank"), "vec_rank": e.get("vec_rank"),
                                  **self._base(e["hit"])}))
            rows.sort(key=lambda x: x[0], reverse=True)
            return self._finalize([d for _, d in rows], k, collapse)

        # (가) hybrid/fusion = RRF 융합 (BM25순위 + 벡터순위)
        rows = []
        for e in pool.values():
            br, vr = e.get("bm25_rank"), e.get("vec_rank")
            rrf = (1.0 / (RRF_K + br) if br else 0.0) + (1.0 / (RRF_K + vr) if vr else 0.0)
            rows.append((rrf, {"score": round(rrf, 6), "retrievers":
                               [x for x in ("bm25" if br else None, "vector" if vr else None) if x],
                               "bm25_rank": br, "vec_rank": vr, **self._base(e["hit"])}))
        rows.sort(key=lambda x: x[0], reverse=True)
        return self._finalize([d for _, d in rows], k, collapse)

    @staticmethod
    def _finalize(rows: list[dict], k: int, collapse: bool) -> list[dict]:
        """근접중복(동일 조 정체성)을 접어 상위 k개만 남기고 rank를 매긴다."""
        out, seen = [], set()
        for d in rows:
            if collapse:
                key = _dedup_key(d)
                if key in seen:
                    continue
                seen.add(key)
            d["rank"] = len(out) + 1
            out.append(d)
            if len(out) >= k:
                break
        return out

    def analyze(self, query: str) -> list[dict]:
        """질의를 *검색용* 분석기(korean_search: 유의어 포함)로 분해.
        토큰마다 형태소 POS 태그(pos)와 유의어 확장 여부(synonym)를 함께 반환."""
        if self.es is None:
            self.load()
        resp = self.es.indices.analyze(
            index=ES_INDEX, analyzer="korean_search", text=query, explain=True)
        detail = resp["detail"]
        toks = (detail["tokenfilters"][-1]["tokens"]
                if detail.get("tokenfilters") else detail["tokenizer"]["tokens"])
        out = []
        for t in toks:
            is_syn = t.get("type") == "SYNONYM"
            out.append({
                "token": t["token"],
                "pos": t.get("leftPOS") or ("유의어(synonym)" if is_syn else t.get("type", "")),
                "synonym": is_syn,
            })
        return out

    def morphemes(self, query: str) -> list[dict]:
        """문장 전체 형태소 분석(조사·어미 포함, POS 필터 적용 전). POS 태그 함께."""
        if self.es is None:
            self.load()
        resp = self.es.indices.analyze(
            index=ES_INDEX, tokenizer="nori_mixed", text=query, explain=True)
        toks = resp["detail"]["tokenizer"]["tokens"]
        return [{"morph": t["token"], "pos": t.get("leftPOS", "")} for t in toks]
