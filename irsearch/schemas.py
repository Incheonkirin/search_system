"""API 요청/응답 스키마(pydantic). Postman/문서에서 타입이 그대로 보인다."""
from __future__ import annotations
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="검색 질의(자연어)")
    k: int = Field(10, ge=1, le=50, description="상위 결과 개수")
    mode: str = Field("rerank", description="lexical | semantic | hybrid | rerank | cross")
    collapse: bool = Field(False, description="근접중복(동일 조 제목) 접기 — 기본 OFF(상품별로 다 보임)")


class SearchHit(BaseModel):
    rank: int
    score: float                      # 모드별 랭킹 점수(hybrid=RRF, 그 외=ES score)
    retrievers: list[str] = []        # 이 결과를 올린 검색기(bm25/vector)
    bm25_rank: int | None = None      # BM25에서의 순위
    vec_rank: int | None = None       # 벡터(kNN)에서의 순위
    doc_id: str
    doc_title: str          # 대제목
    gwan: str = ""          # 관(款)
    section: str            # 소제목
    section_path: str       # "대제목 > 관 > 소제목"
    product_type: str
    insurer: str
    chunk_id: str
    snippet: str            # 하이라이트 미리보기(검색어 주변)
    text: str = ""          # 조(條) 전체 본문


class Morph(BaseModel):
    morph: str
    pos: str = ""            # 형태소 POS 태그(nori)


class Token(BaseModel):
    token: str
    pos: str = ""            # 형태소 POS 태그(nori). 예: NNG(General Noun)
    synonym: bool = False    # 유의어로 확장된 토큰인지


class SearchResponse(BaseModel):
    query: str
    mode: str = "hybrid"             # lexical | semantic | hybrid
    morphemes: list[Morph] = []      # 전체 문장 형태소(조사·어미 포함)
    search_terms: list[Token] = []   # 최종 검색어(POS 필터·유의어 적용)
    count: int
    results: list[SearchHit]
