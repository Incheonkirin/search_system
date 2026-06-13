"""FastAPI 검색 서버 (진입점).

실행:  cd search_system && uvicorn server:app --host 127.0.0.1 --port 8800 --reload
Postman:
  GET  http://127.0.0.1:8800/search?q=암 진단보험금&k=5
  POST http://127.0.0.1:8800/search   body(JSON): {"query":"보험료 납입 연체", "k":5}
"""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.requests import Request

from irsearch.config import STATS_PATH
from irsearch.indexer import build_index
from irsearch.logconf import setup_logging
from irsearch.search import Searcher
from irsearch.schemas import SearchRequest, SearchResponse

log = setup_logging()

app = FastAPI(title="보험 약관 IR 검색", version="0.1.0",
              description="메트라이프 공개 약관 BM25 검색 (검색식은 추후 튜닝)")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    rid = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("http_request", extra={"fields": {
        "rid": rid,
        "method": request.method,
        "path": request.url.path,
        "query": dict(request.query_params),     # 디코딩된 질의(한글 그대로)
        "status": response.status_code,
        "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
        "client": request.client.host if request.client else None,
    }})
    return response


searcher = Searcher()


@app.on_event("startup")
def _startup():
    try:
        searcher.load()
    except Exception as e:
        print(f"[warn] ES 준비 안 됨: {e} — `make es-up` 후 `make index` 하세요.")


_UI_HTML = Path(__file__).parent / "ui.html"


@app.get("/")
def ui():
    """검색 웹 UI — 쿼리→형태소분석→IR(BM25)·시맨틱(임베딩) 검색→RRF·리랭크."""
    return FileResponse(_UI_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return json.loads(STATS_PATH.read_text(encoding="utf-8")) if STATS_PATH.exists() else {}


@app.get("/analyze")
def analyze(q: str = Query(..., description="형태소 분석할 질의")):
    return {"query": q,
            "morphemes": searcher.morphemes(q),       # 전체 문장 형태소
            "search_terms": searcher.analyze(q)}      # 최종 검색어


@app.get("/search", response_model=SearchResponse)
def search_get(q: str = Query(..., description="검색 질의"),
               k: int = Query(10, ge=1, le=50),
               mode: str = Query("rerank", description="lexical|semantic|hybrid|rerank|cross"),
               collapse: bool = Query(False, description="근접중복(동일 조 제목) 접기 — 기본 OFF")):
    t0 = time.perf_counter()
    res = searcher.search(q, k, mode, collapse)
    terms = searcher.analyze(q)
    log.info("search", extra={"fields": {
        "query": q, "mode": mode, "k": k, "collapse": collapse,
        "terms": [t["token"] for t in terms],
        "hits": len(res), "took_ms": round((time.perf_counter() - t0) * 1000, 1)}})
    return {"query": q, "mode": mode, "morphemes": searcher.morphemes(q),
            "search_terms": terms, "count": len(res), "results": res}


@app.post("/search", response_model=SearchResponse)
def search_post(req: SearchRequest):
    t0 = time.perf_counter()
    res = searcher.search(req.query, req.k, req.mode, req.collapse)
    terms = searcher.analyze(req.query)
    log.info("search", extra={"fields": {
        "query": req.query, "mode": req.mode, "k": req.k, "collapse": req.collapse,
        "terms": [t["token"] for t in terms],
        "hits": len(res), "took_ms": round((time.perf_counter() - t0) * 1000, 1)}})
    return {"query": req.query, "mode": req.mode, "morphemes": searcher.morphemes(req.query),
            "search_terms": terms, "count": len(res), "results": res}


@app.post("/reindex")
def reindex():
    s = build_index()
    searcher.load()
    return {"reindexed": True, **s}
