# 오지 레스토랑 리뷰 서버

기존 FastAPI 리뷰 과제에 LangChain과 LangGraph를 연결했습니다. 리뷰를 등록하면 매장 규정을 찾아 사장님 답글을 작성하고 검토합니다. 통과한 답글을 저장해 리뷰 조회에서 읽거나 한 줄 지시로 수정할 수 있습니다.

## 로컬 실행

Python 3.14와 uv가 필요합니다. 아래는 프로젝트 루트에서 실행하는 PowerShell 명령입니다.

```powershell
uv sync --locked
uv run python -m app.services.policy_index
uv run python -m app.services.policy_index --evaluate

$env:FAKE_LLM = "true"
uv run fastapi run --host 127.0.0.1 --port 8000
```

첫 인덱스 구축은 공개 다국어 임베딩 모델을 다운로드합니다. 강의의 `data/ohgi_regulations.md`를 35개 조문으로 나눠 Chroma에 저장합니다. 앱 시작 시 준비된 인덱스와 모델 캐시를 로컬에서 읽습니다. 규정이 바뀌면 인덱스를 다시 만든 뒤 앱을 재시작합니다. 기본 데이터 경로는 `.local/`입니다.

[Swagger UI](http://127.0.0.1:8000/docs)에서 등록·조회·답글·설정 API를 호출할 수 있습니다. 가짜 모드의 그래프는 정해진 예문과 판정을 사용합니다. `/reply/sync`는 과제의 프롬프트 반향 응답을 유지합니다. `FAKE_LLM=true`를 사용하면 키 설정과 관계없이 OpenAI 비용 없이 실습할 수 있습니다.

## API

| 메서드 | 경로 | 결과 |
|---|---|---|
| POST | `/api/reviews` | 리뷰 등록 201 |
| GET | `/api/reviews`, `/api/reviews/{id}` | 목록과 개별 리뷰 조회 |
| POST | `/api/reviews/{id}/reply` | 검색·작성·검토 후 구조화 결과 |
| POST | `/api/reviews/{id}/reply/stream` | 같은 그래프의 진행·초안 조각·최종 결과 |
| POST | `/api/reviews/{id}/reply/revise` | `instruction`으로 직전 저장 답글 수정 |
| POST | `/api/reviews/{id}/reply/sync` | 동기 LCEL 실습 경로. `reviewId`, `reply`, `model` 반환 |
| GET, PUT | `/api/reply-preferences` | 매장 말투와 문장 상한 읽기·저장 |

리뷰 등록 키는 `customerName`(1~20자), `rating`(1~5 정수), `content`(1~500자)입니다. 리뷰 1~3은 서버가 시작할 때 생성됩니다. 답글 결과에는 `status`, `reply`, `tone`, `needsFollowup`, `articleIds`, `attempts`가 들어갑니다.

- `status=ready`이면 서버가 초안을 저장합니다. `needsFollowup=true`는 주문이나 사진을 매장에서 추가 확인해야 한다는 뜻입니다.
- `status=needs_review`이면 `reply=null`이고, 앞서 저장한 답글이 있다면 유지됩니다.
- 404는 없는 리뷰, 409는 같은 리뷰를 이미 처리 중이거나 수정할 답글이 없는 경우입니다. JSON 응답에서 모델 실패는 502입니다.
- SSE는 응답 전 404·409를 HTTP로, 응답 시작 뒤 실패를 `error` 이벤트로 전달합니다. `draft_delta`는 검토 중인 글이고 `attempt`가 바뀌면 미리보기를 새로 시작합니다. 저장 결과는 `final`에서 확인합니다. POST는 `fetch`의 스트림이나 `curl.exe -N`으로 읽을 수 있습니다.

### 복합 리뷰 실습

서버를 실행한 뒤 다른 PowerShell 창에서 아래를 실행합니다.

```powershell
$base = "http://127.0.0.1:8000"
$body = @{
    customerName = "실습 손님"
    rating = 1
    content = "주문한 메뉴와 다른 음식이 왔고 배달 용기가 깨져 국물이 쏟아졌어요."
} | ConvertTo-Json
$review = Invoke-RestMethod -Method Post -Uri "$base/api/reviews" `
    -ContentType "application/json; charset=utf-8" -Body $body
$reviewUrl = "$base/api/reviews/$($review.reviewId)"
curl.exe -N -X POST "$reviewUrl/reply/stream"
Invoke-RestMethod -Uri $reviewUrl
Invoke-RestMethod -Method Post -Uri "$reviewUrl/reply/revise" `
    -ContentType "application/json; charset=utf-8" `
    -Body (@{ instruction = "한 문장으로 줄여 주세요." } | ConvertTo-Json)
```

첫 초안에서 빠진 배달 사고 조건이 검토에서 발견됩니다. 가짜 모드의 두 번째 초안에는 제12조의 24시간과 제17조의 1시간, 각각의 사진 제출이 들어갑니다. 이어서 한 문장 수정에서도 두 기한을 확인할 수 있습니다.

### 매장 선호

```powershell
Invoke-RestMethod -Method Put -Uri "$base/api/reply-preferences" `
    -ContentType "application/json; charset=utf-8" `
    -Body (@{ style = "따뜻"; maxSentences = 3 } | ConvertTo-Json)
```

`style`은 `정중` 또는 `따뜻`, `maxSentences`는 1~5입니다. SQLite Store에 보관해 재시작 뒤에도 이 값을 사용합니다. 리뷰, 답글과 리뷰별 체크포인트는 프로세스 메모리에 있으므로 재시작하면 초기화됩니다. 실습 서버는 한 매장·한 프로세스를 전제로 합니다.

## 구현 순서

| 단계 | 구현 | 위치 |
|---|---|---|
| 1 | `ChatPromptTemplate → ChatOpenAI → StrOutputParser` LCEL | `app/clients/llm_client.py` |
| 2 | Pydantic 스키마로 초안과 검토 결과를 파싱 | `app/schemas/reply_draft.py` |
| 3 | `retrieve → draft → review` 후 미통과 시 한 번 재작성 | `app/services/reply_workflow.py` |
| 4 | `astream()`의 진행, 초안 조각, 완료 상태를 SSE로 변환 | `app/services/reply_service.py`, `app/routers/review_router.py` |
| 5 | 리뷰별 `thread_id` 체크포인트와 SQLite 매장 선호 Store | `app/services/reply_service.py`, `app/main.py` |
| 6 | 필수 조문 ID 조회 + Chroma 벡터·BM25 검색과 RRF 순위 결합 | `app/services/policy_search.py`, `app/services/hybrid_retriever.py`, `app/services/policy_index.py` |

초안은 최대 2회입니다. 그래프 단계 한도는 8, 비동기 요청 시간 상한 기본값은 60초입니다. 초안과 검토가 모두 모델을 쓰는 경우 최대 4번의 모델 작업이 일어납니다. SDK 재시도는 호출별 연결 장애 처리로 별도 적용합니다. 규정 인덱스가 준비되지 않으면 규정 근거가 필요한 요청을 매장 확인으로 보냅니다.

## 검색 평가

강의 골든셋 20개에서 정답 조문이 앞의 한 개 또는 세 개 후보에 들어오는지 측정했습니다. 이 평가는 답글 생성 전에 실행하며, 필수 조문 ID 조회는 순위 비교에서 제외합니다.

| 검색 | Hit@1 | Hit@3 |
|---|---:|---:|
| Chroma 벡터 | 13/20 | 15/20 |
| BM25 | 14/20 | 20/20 |
| 하이브리드 RRF | 17/20 | 19/20 |

하이브리드는 이 실습 자료에서 첫 후보를 더 자주 맞혔고, 주류 배달 질문은 BM25의 세 후보에만 정답이 들었습니다. 실제 요청에서는 오배송·이물질·배달 사고 등 적용 규정이 확인된 경우 정확한 조문 ID를 우선 확보합니다. 범용 매장 FAQ의 생성 정확도는 이 검색 결과로 판단할 수 없습니다.

## 테스트와 동시 요청

```powershell
$env:FAKE_LLM = "true"
uv run pytest -q
```

테스트는 가짜 모델과 `httpx.MockTransport`를 사용합니다. 모의 전송으로 실제 `ChatOpenAI`의 구조화 파싱과 토큰 조각, 끊긴 응답·실패 응답을 확인합니다.

부하 실습은 가짜 모델 로컬 서버를 8000번에 띄운 상태에서 `uv run python benchmark/benchmark.py`를 실행합니다. 도구는 `/health`로 모델을 확인하고 시험용 리뷰 100개를 등록합니다. 두 경로의 응답 시간(p50·p95), 처리량(RPS), 실패 건수를 비교합니다. 동기 체인과 비동기 그래프에는 서로 다른 작업량이 포함됩니다.

리팩토링 전 OpenAI 호출 결과와 동기·비동기 측정치는 기준 커밋의 README에 보존되어 있습니다. `git show d13fba9:README.md`
