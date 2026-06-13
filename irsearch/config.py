"""전역 설정 — 경로·서버·색인 파라미터를 한곳에서 관리."""
from pathlib import Path

PKG = Path(__file__).resolve().parent       # irsearch/
ROOT = PKG.parent                            # search_system/
DATA = ROOT / "data"
CORPUS_DIR = DATA / "corpus"                 # 문서 원본(.txt)
INDEX_DIR = DATA / "index"                   # 색인 산출물
INDEX_PATH = INDEX_DIR / "index.pkl"
STATS_PATH = INDEX_DIR / "stats.json"

# 서버
HOST = "127.0.0.1"
PORT = 8800

# 코퍼스/색인
N_DOCS = 100                                 # 시드 코퍼스 문서 수
CHUNK_MAX_CHARS = 8000                        # 청크 = 조(條) 통째. 이 한도 넘는 초장문 조만 항 경계로 분할
DEFAULT_TOP_K = 10

# Elasticsearch
ES_URL = "http://localhost:9200"
ES_INDEX = "yakgwan"

# 시맨틱 검색(임베딩)
EMBED_MODEL = "BAAI/bge-m3"      # 한국어/다국어 검색 SOTA
EMBED_DIM = 1024                 # bge-m3 임베딩 차원
EMBED_BATCH = 32                 # 색인 임베딩 배치 크기
RRF_K = 60                       # RRF 융합 상수(표준값)
