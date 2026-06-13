# 보험 약관 IR 검색 (Korean insurance-clause retrieval)

한국어 보험 약관 조항을 검색하는 정보검색(IR) 시스템. 형태소 분석(nori) 위에
BM25(어휘) · BGE-M3 임베딩(시맨틱) · RRF 융합 · 크로스인코더 리랭크를 얹어
조(條) 단위로 검색한다. Elasticsearch 백엔드, FastAPI 서버.

## 구성

- **형태소 분석**: Elasticsearch `nori` (질의·문서 토큰화)
- **어휘 검색**: BM25
- **시맨틱 검색**: `BAAI/bge-m3` 임베딩 kNN
- **융합**: RRF(Reciprocal Rank Fusion)
- **리랭크**: 크로스인코더
- **청크 단위**: 약관 조(條) 통째 (초장문 조만 항 경계로 분할)

검색 모드: `lexical | semantic | hybrid | rerank | cross`

## 실행

```bash
# 1) Elasticsearch(nori 포함) 기동
make es-up

# 2) 약관 .txt 코퍼스를 data/corpus/ 에 둔다 (형식은 아래)
#    그리고 색인
make index

# 3) 서버
uvicorn server:app --host 127.0.0.1 --port 8800 --reload
```

웹 UI: http://127.0.0.1:8800/  ·  API: `GET /search?q=암 진단보험금&k=5&mode=rerank`

## 코퍼스

약관 `.txt` 파일을 `data/corpus/` 에 넣고 `make index`. 약관 원본은 저장소에 포함하지 않는다.
조(條) 단위로 청킹되며 `### 제N조(...)` 헤더 구조를 인식한다.

## API

| 엔드포인트 | 설명 |
|---|---|
| `GET /search?q=&k=&mode=&collapse=` | 검색 |
| `POST /search` | 검색 (JSON body: `{query, k, mode, collapse}`) |
| `GET /analyze?q=` | 형태소 분석 결과만 |
| `GET /stats` | 색인 통계 |
| `GET /health` | 헬스체크 |
| `POST /reindex` | 재색인 |

## 라이선스

코드: 본인 작성. 약관 코퍼스는 미포함(저작권).
