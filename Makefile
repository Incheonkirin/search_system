.PHONY: setup es-up es-down es-logs index serve clean

setup:        ## 파이썬 의존성 설치
	pip install -r requirements.txt

es-up:        ## Elasticsearch(+nori) 컨테이너 빌드·기동
	docker-compose up -d --build

es-down:      ## ES 컨테이너 중지
	docker-compose down

es-logs:      ## ES 로그
	docker-compose logs -f es

index:        ## 코퍼스 → ES 색인 (ES 떠 있어야 함)
	python -m irsearch.indexer

serve:        ## 검색 API 서버 (:8800)
	uvicorn server:app --host 127.0.0.1 --port 8800 --reload

clean:        ## 로컬 산출물 정리(코퍼스 유지)
	rm -rf data/index data/pdf
