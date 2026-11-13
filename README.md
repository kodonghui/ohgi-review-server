# 오지 레스토랑 리뷰 서버

리뷰에 담긴 질문을 나누고 매장 규정의 근거를 찾은 뒤, 사장님 답글을 작성·검토하는 FastAPI 실습입니다. 통과한 답글은 리뷰에 저장하며 JSON, SSE, 한 문장 수정 경로에서 같은 LangGraph를 사용합니다.

| 검증한 동작 | 초기 → 최종 |
| --- | ---: |
| 35조 규정집 · 필요한 조문을 모두 포함해 저장 | 6/44 → 43/44 |
| 50페이지·155조 규정집 · 새 조문의 번호와 핵심 구절까지 저장 | 0/24 → 24/24 |
| 155조 대리 수령 등 적용 범위 진단 · 번호와 핵심 구절까지 저장 | 5/11 → 11/11 |
| 같은 진단의 실제 상태·대상 불명 문의 · 올바르게 보류 | 2/5 → 5/5 |
| 별도 픽업·인계 완료 표현 · 올바르게 보류 | 0/3 → 3/3 |

[본문의 버전별 벤치마크](#벤치마크) · [35조 실행 기록](benchmark/RESULTS.md) · [155조 실행 기록](benchmark/scale/VERSIONS.md) · [벤치마크 커밋 대조표](benchmark/SOURCES.md)

## 로컬 실행

Python 3.14와 uv가 필요합니다. 다음은 Windows PowerShell에서 저장소 루트로 이동해 실행하는 명령입니다.

```powershell
uv sync --locked
uv run --locked python -m app.services.policy_index
$env:FAKE_LLM = 'true'
$env:LANGSMITH_TRACING = 'false'
uv run --locked fastapi run --host 127.0.0.1 --port 8000
```

첫 인덱스 구축 때 다국어 MiniLM 임베딩 모델을 내려받습니다. 기본 앱은 [실습 규정집 35조](data/ohgi_regulations.md)와 `.local/`을 사용합니다. 앱 시작 전에 인덱스를 만들고 규정이 바뀌면 인덱스를 다시 만든 뒤 서버를 재시작합니다. 가짜 모델 모드로 OpenAI 키 없이 전체 흐름을 실행할 수 있습니다.

[Swagger UI](http://127.0.0.1:8000/docs)에서 리뷰 등록·답글 생성·조회·수정을 실행할 수 있습니다. 155조와 제69조 대리 수령은 아래 확장판 재현 명령으로 실행합니다.

## 아키텍처

```mermaid
flowchart TD
    A["FastAPI · 등록된 리뷰의 답글 요청"] --> R["질문 분리와 적용 규정 확인"]
    R -->|"개별 주문 상태 등"| H["needs_review · 직원 확인"]
    R -->|"등록된 사건 규정"| ID["필수 조문 ID 조회"]
    R -->|"문서 검색"| S["Chroma 벡터 + BM25 → RRF 후보"]
    ID --> E["질문별 적용 범위와 근거 문장 확인"]
    S --> E
    E -->|"일부 근거 부족"| Q["해당 질문만 한 번 재검색 후 확인"]
    Q -->|"근거 부족"| H
    E -->|"근거 확보 또는 일반 후기"| D["구조화 답글 초안"]
    Q -->|"근거 확보"| D
    D --> V["필수 조문·조건 확인 및 모델 검토"]
    V -->|"첫 실패"| D
    V -->|"두 번째 실패"| H
    V -->|"통과"| P["ready · 답글 저장"]
```

- **검색** · Chroma는 임베딩으로 의미가 가까운 조문을, BM25는 한국어 단어와 글자 쌍이 겹치는 조문을 찾습니다. 각각 최대 8개 후보를 RRF로 정렬하고, 상위 몇 개 밖에 있는 조문도 직접 근거를 찾을 때 사용합니다. [검색기](app/services/hybrid_retriever.py) · [질문별 검색](app/services/policy_search.py)
- **근거 선택** · ‘포장 음식을 대신 받는다’와 ‘분실물을 대신 받는다’는 행동이 같아도 적용 대상이 다릅니다. 제목과 본문에서 대상을 확인하고, 그 행동의 조건을 설명하는 문장을 고릅니다. 제69조는 질문에 주문번호가 없어도 안내할 수 있습니다. 실제로 누군가 받아 갔는지는 매장 기록이 필요합니다. [문장 선택](app/services/policy_passages.py)
- **작성과 검토** · 그래프는 필요한 조문이 확보됐을 때 초안을 작성합니다. 사건별 필수 안내와 모델의 구조화 판정을 통과하면 `ready`를 반환합니다. 최대 초안 2회, 그래프 단계 한도 8, 비동기 요청 시간 기본 60초입니다. [그래프](app/services/reply_workflow.py)
- **상태** · `InMemorySaver`가 `review:{id}`별 체크포인트를 관리합니다. SQLite Store의 `("stores", "ohgi", "reply")` 네임스페이스에는 매장 말투와 문장 상한을 저장합니다. 설정은 재시작 후 다시 읽으며 리뷰·답글·체크포인트는 메모리에서 초기화됩니다. [앱 초기화](app/main.py) · [답글 서비스](app/services/reply_service.py)

### 실습 1–6단계

| 단계 | 구현 |
| --- | --- |
| 1 | [LCEL 체인](app/clients/llm_client.py)의 `ChatPromptTemplate → ChatOpenAI → StrOutputParser` |
| 2 | [구조화 응답](app/schemas/reply_draft.py)의 Pydantic 초안·검토 스키마 |
| 3 | [LangGraph](app/services/reply_workflow.py)의 `retrieve → draft → review`와 제한된 재작성 |
| 4 | [SSE](app/routers/review_router.py)의 진행·초안 조각·최종 상태 |
| 5 | [체크포인터와 Store](app/main.py)의 리뷰별 상태·매장 선호·답글 수정 |
| 6 | [규정 RAG](app/services/policy_index.py)의 사전 인덱싱, 하이브리드 검색, 질문별 원문 검증 |

## API

| 메서드 | 경로 | 결과 |
| --- | --- | --- |
| POST | `/api/reviews` | 리뷰 등록 201 |
| GET | `/api/reviews`, `/api/reviews/{id}` | 목록과 개별 리뷰 조회 |
| POST | `/api/reviews/{id}/reply` | 구조화 답글 결과 |
| POST | `/api/reviews/{id}/reply/stream` | SSE 진행·초안 조각·최종 결과 |
| POST | `/api/reviews/{id}/reply/revise` | `instruction`으로 직전 저장 답글 수정 |
| POST | `/api/reviews/{id}/reply/sync` | 동기 LCEL 실습 경로 |
| GET, PUT | `/api/reply-preferences` | 매장 말투와 문장 상한 읽기·저장 |

등록 키는 `customerName`(1~20자), `rating`(1~5 정수), `content`(1~500자)입니다. 답글 결과에는 `status`, `reply`, `tone`, `needsFollowup`, `articleIds`, `attempts`가 포함됩니다. `ready`는 저장을 뜻합니다. `needs_review`의 `reply`는 `null`이며 기존 답글을 유지합니다. `needsFollowup`은 매장에서 추가 확인할 내용이 있다는 표시입니다.

404는 없는 리뷰, 409는 같은 리뷰의 처리 중 충돌 또는 수정할 답글이 없는 경우입니다. JSON 경로는 모델 실패를 502로 반환하고 SSE는 전송을 시작한 뒤의 실패를 `error` 이벤트로 보냅니다. `draft_delta`는 임시 텍스트이며 `attempt`가 달라지면 미리보기를 새로 시작합니다. 저장 여부는 `final`에서 확인합니다.

### 복합 리뷰 실습

서버를 실행한 뒤 다른 PowerShell 창에서 실행합니다. 가짜 모델의 두 번째 초안과 한 문장 수정에서 제12조의 24시간과 제17조의 1시간 및 사진 조건을 확인할 수 있습니다.

```powershell
$base = 'http://127.0.0.1:8000'
$body = @{
    customerName = '실습 손님'
    rating = 1
    content = '주문한 메뉴와 다른 음식이 왔고 배달 용기가 깨져 국물이 쏟아졌어요.'
} | ConvertTo-Json
$review = Invoke-RestMethod -Method Post -Uri "$base/api/reviews" `
    -ContentType 'application/json; charset=utf-8' -Body $body
$reviewUrl = "$base/api/reviews/$($review.reviewId)"
curl.exe -N -X POST "$reviewUrl/reply/stream"
Invoke-RestMethod -Uri $reviewUrl
Invoke-RestMethod -Method Post -Uri "$reviewUrl/reply/revise" `
    -ContentType 'application/json; charset=utf-8' `
    -Body (@{ instruction = '한 문장으로 줄여 주세요.' } | ConvertTo-Json)
```

`PUT /api/reply-preferences`에 `{"style":"따뜻","maxSentences":3}`을 보내 매장 선호를 바꿀 수 있습니다. 말투는 `정중` 또는 `따뜻`, 문장 상한은 1~5입니다.

## 벤치마크

API 실험은 Windows의 FastAPI `TestClient`에서 문항별로 3회씩 순차 실행했습니다. `POST /reply`가 반환한 답글과 `GET /api/reviews/{id}`에서 읽은 저장값을 대조했습니다. 기존 44문항은 필요한 조문 번호가 전부 있는지, 확장판의 새 24문항은 핵심 구절까지 답글에 포함했는지 확인했습니다. p50과 p95는 `POST /reply` 지연의 중앙값과 95번째 백분위입니다.

### 검색기 비교 · 강의 골든셋 20문항

Hit@1은 첫 후보, Hit@3은 상위 세 후보에 정답 조문이 들어온 문항 수입니다.

| 검색기 | Hit@1 | Hit@3 |
| --- | ---: | ---: |
| Chroma 벡터 | 13/20 | 15/20 |
| BM25 | 14/20 | 20/20 |
| 하이브리드 RRF | 17/20 | 19/20 |

검색기 단독 평가 명령은 `uv run --locked python -m app.services.policy_index --evaluate`입니다.

### 기본 규정 35조 · 답글 API의 개선 과정

같은 35조 규정집과 같은 52문항으로 코드 5개를 비교했습니다. 조문 평가 대상은 44문항이며 그중 8건이 복합 문의입니다.

| 변경 · 소스와 원본 | 필수 조문 확보 | 필수 조문 포함 저장 | 복합 문의 저장 | 조문 누락 저장 | API p50 / p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기준 [`f678b2c`](benchmark/results/baseline.json) | 7/44 | 6/44 | 1/8 | 36건 | 4.55 / 16.86 |
| 질문마다 근거 확인 [`414f3af`](benchmark/results/question-routing.json) | 36/44 | 36/44 | 4/8 | 0건 | 13.58 / 25.76 |
| 근거가 부족한 질문만 재검색 [`d016d1c`](benchmark/results/evidence-retry.json) | 43/44 | 43/44 | 8/8 | 0건 | 13.80 / 24.08 |
| 사건별 안내 검증 [`cbd8239`](benchmark/results/verified-notices.json) | 43/44 | 43/44 | 8/8 | 0건 | 16.22 / 27.79 |
| 개별 상태·부정 표현 보완 [`a189167`](benchmark/results/routing-corrections.json) | 43/44 | 43/44 | 8/8 | 0건 | 16.99 / 30.84 |

- **조문이 빠진 저장 36건 → 0건.** 질문을 나누고 각각의 근거를 요구했습니다. 매장 확인이 필요한 7건 중 잘못 저장하던 5건도 보류했습니다.
- **복합 문의 저장 4/8 → 8/8.** 52건 중 7건에서 근거가 부족한 하위 질문만 다시 검색했습니다. ‘맥주도 배달돼?’는 처음 RRF 상위 3개에서 빠진 제18조를 넓힌 후보에서 찾아 답글에 반영했습니다.
- **사건별 안내 검증.** 오배송 접수 24시간과 용기 파손 접수 1시간을 서로 바꿔 쓴 초안은 초기 버전에서 검토를 통과해 저장됐습니다. 사건별 검증을 추가하자 초안 두 번이 모두 걸렸고 기존 답글을 유지했습니다. 개별 상태와 부정 표현을 보완한 버전에서는 실제 환불 입금 문의를 직원 확인으로 보내고, 오배송이 없었다는 파손 신고에는 제17조만 적용했습니다.

43/44에서 남은 G15는 중증 땅콩 알레르기가 있는 손님의 주문 안전 문의입니다. 특정 주문의 안전 판단은 직원 확인으로 보냈습니다. 제20조 원문을 보류 상태에 붙이는 별도 시제품은 **근거 확보 44/44, 답글 저장 43/44**였습니다. [35조 상세 기록](benchmark/RESULTS.md) · [제20조 시제품](benchmark/experiments/ANSWERABILITY.md)

### 같은 코드로 9페이지·35조와 50페이지·155조 비교

기존 35조에 가상 규정 120조를 더하고, 같은 코드 `b3c7dba`에 같은 기존 문항을 넣었습니다. 검색 순위와 후보 합집합은 원질문을 한 번 검색했을 때 필요한 조문을 전부 찾은 건수입니다.

| 기존 문항의 지표 | 9페이지·35조 | 50페이지·155조 |
| --- | ---: | ---: |
| Chroma 상위 3개 | 33/44 | 28/44 |
| BM25 상위 3개 | 40/44 | 31/44 |
| 하이브리드 RRF 상위 3개 | 38/44 | 33/44 |
| Chroma·BM25 전체 후보 합집합 | 44/44 | 40/44 |
| 답글 API의 필수 조문 포함 저장 | 43/44 | 43/44 |
| 재검색이 발생한 요청 | 7/52 | 10/52 |
| 답글 API p50 / p95 (ms) | 13.07 / 24.39 | 14.53 / 26.28 |
| 인덱스 구축 시간 | 2.54초 | 8.41초 |

35조에서는 BM25가 상위 3개에 필요한 조문을 가장 자주 넣었고, 155조에서는 RRF가 가장 자주 넣었습니다. 후보 합집합의 확보 건수가 44/44에서 40/44로 내려갔을 때도, API는 질문별 근거 확인과 재검색을 통해 43/44를 저장했습니다. 35조의 별도 검색 실험에서 로컬 cross-encoder로 후보를 재정렬하면 상위 3개에서 42/44를 확보했고, 재정렬에 중앙값 767.4ms가 추가됐습니다. 현재 API는 두 검색기의 후보를 유지하고, 필요한 조문을 질문별로 확인해 빠진 질문만 다시 검색합니다. [규정집 크기 비교](benchmark/scale/RESULTS.md) · [재정렬 실험](benchmark/RESULTS.md#하이브리드와-재정렬을-고른-근거)

### 50페이지·155조 · 고정 84문항의 버전 비교

같은 155조에서 코드만 바꿨습니다. 기존 52문항에 새 조문 단독 18건, 기존·새 조문을 함께 묻는 6건, 규정만으로 답할 수 없는 7건, 일반 후기 1건을 더한 시험지입니다.

| 변경 | 소스·문항별 원본 | 기존 필수 조문 포함 저장 | 새 조문·핵심 구절 저장 | 대조군 보류 | 새 조문 API p50 / p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| 기존 규칙 라우터 | [`94cc0df`](benchmark/scale/results/155-ce33186.json) | 43/44 | 0/24 | 3/7 | 7.38 / 41.78 |
| 상태·비공개 문의 보류 | [`a3f111a`](benchmark/scale/results/155-cc0b0aa.json) | 43/44 | 0/24 | 7/7 | 7.93 / 41.24 |
| 새 조문 원문 선택 | [`9038d9a`](benchmark/scale/results/155-2a4a8fb.json) | 43/44 | 24/24 | 7/7 | 28.65 / 75.87 |
| 겹치는 질문 보완 | [`2695f60`](benchmark/scale/results/155-83a8774.json) | 43/44 | 24/24 | 7/7 | 49.74 / 117.57 |
| 조문의 적용 대상과 조건 확인 | [`c6f0fb0`](benchmark/scale/results/155-5993846.json) | 43/44 | 24/24 | 7/7 | 31.04 / 73.78 |
| 픽업·인계 완료 상태 확인 | [`c7052de`](benchmark/scale/results/155-c7052de.json) | 43/44 | 24/24 | 7/7 | 53.51 / 120.97 |

- **검색 후보에 근거 23/24, 초기의 새 조문·구절 저장 0/24.** 휠체어로 지하 주차장에서 올라가는 길을 물었을 때, 기존 라우터는 제36조를 후보에 두고도 ‘주차’라는 표현에 따라 제4조의 주차 요금을 저장했습니다. 질문에 직접 답하는 구절을 찾도록 바꾼 뒤에는 제36조의 A구역 승강기를 안내했습니다.
- **새 조문 단독 18/18, 복합 문의 6/6.** 유아 의자·예약 지연·주류 배달·배달 마감을 함께 물은 S24에서 원질문 검색만으로는 네 조문이 다 나오지 않았습니다. 하위 질문별로 근거를 대조하고 빠진 근거를 보완해 제37·49·18·15조를 모두 반영했습니다. 조문이 빠진 채 저장한 새 문항은 3건에서 0건으로 줄었습니다.
- **대조군 보류 3/7 → 7/7.** 실제 환불 입금 여부, 차량 주차 지원 등록, 분실물 습득 여부, 관리자 암호처럼 규정집으로 확인할 수 없는 요청을 보류했습니다. 일반 감사 후기는 여섯 버전에서 모두 답글을 저장했습니다.

초기 두 버전은 새 조문 요청 24건 중 21건을 초안 전에 보류했습니다. 이후 버전은 근거 확인부터 작성·검토·저장까지 처리했고, 최종 버전의 새 조문 API 지연은 **p50 53.51ms, p95 120.97ms**였습니다. 한 PC에서 순차 실행한 값입니다. [155조 실험 조건과 문항](benchmark/scale/VERSIONS.md)

### 고정 문항 24/24 이후 발견한 실패

| 진단 세트 | 소스·문항별 원본 | 조문·핵심 구절 저장 | 올바른 보류 |
| --- | --- | ---: | ---: |
| 추가 14문항 | [`94cc0df`](benchmark/scale/results/155-ce33186-diagnostic.json) | 0/10 | 3/4 |
| 추가 14문항 | [`9038d9a`](benchmark/scale/results/155-2a4a8fb-diagnostic.json) | 7/10 | 4/4 |
| 추가 14문항 | [`2695f60`](benchmark/scale/results/155-83a8774-diagnostic.json) | 9/10 | 4/4 |
| 추가 14문항 | [`c6f0fb0`](benchmark/scale/results/155-5993846-diagnostic.json) | 10/10 | 4/4 |
| 추가 14문항 | [`c7052de`](benchmark/scale/results/155-c7052de-diagnostic.json) | 10/10 | 4/4 |
| 적용 범위·상태 16문항 | [`2695f60`](benchmark/scale/results/155-83a8774-scope.json) | 5/11 | 2/5 |
| 적용 범위·상태 16문항 | [`c6f0fb0`](benchmark/scale/results/155-5993846-scope.json) | 11/11 | 5/5 |
| 적용 범위·상태 16문항 | [`c7052de`](benchmark/scale/results/155-c7052de-scope.json) | 11/11 | 5/5 |
| 픽업·인계 완료 4문항 | [`c6f0fb0`](benchmark/scale/results/155-c6f0fb0-pickup-state.json) | 1/1 | 0/3 |
| 픽업·인계 완료 4문항 | [`c7052de`](benchmark/scale/results/155-c7052de-pickup-state.json) | 1/1 | 3/3 |

- **한 요청의 두 예약 질문.** 기존에는 단체 예약 제30조만 답했습니다. 각각의 질문을 유지한 뒤 단체 메뉴 제출 기한인 전날 오후 6시(제51조)와 일반 2인 예약 가능 기간인 다음 14일(제48조)을 함께 저장했습니다.
- **포장과 분실물의 대리 수령.** 제69조는 이미 후보에 있었지만 조건 문장의 점수가 분실물 제43조와 비슷해 보류됐습니다. 조문 전체에서 대상을 확인하고 조건 문장을 고르도록 바꿨습니다. 질문에 주문번호를 적지 않아도 제69조의 주문번호 전달 조건을 안내하고, 분실물에는 제43조의 이름 확인을 안내합니다. 16문항에서 실제 상태를 확인해야 할 문의를 잘못 저장한 건은 3건에서 0건으로 줄었습니다.
- **‘픽업해도 되나요?’와 ‘이미 픽업했나요?’.** 별도 진단의 일반 조건 1건에는 제69조를 붙여 저장했습니다. 픽업이나 인계가 실제 완료됐는지 묻는 3건은 초안 작성 전에 보류했습니다.

### 벤치마크 재현

기본 35조 인덱스와 임베딩 캐시를 준비한 상태에서 실행합니다.

```powershell
uv run --locked python benchmark/policy_quality.py --output .local/benchmark/policy-35.json
```

50페이지 PDF 생성기는 Windows 맑은 고딕을 사용합니다. 확장판 벤치와 HTTP 점검은 선택한 Git 커밋을 임시 경로에 풀어 해당 규정집을 인덱싱합니다.

```powershell
$book = '.local/benchmark/regulations-50p.md'
$version = 'c7052de'
uv run --no-project --with reportlab python benchmark/scale/render_book.py --markdown $book --pdf .local/benchmark/regulations-50p.pdf
uv run --locked python benchmark/scale/run.py --expanded-only --source-commit $version --markdown $book --output .local/benchmark/router-155.json
uv run --locked python benchmark/scale/run.py --diagnostics-only --source-commit $version --markdown $book --output .local/benchmark/router-155-diagnostics.json
uv run --locked python benchmark/scale/run.py --diagnostic-file benchmark/scale/policy-scope.json --source-commit $version --markdown $book --output .local/benchmark/router-155-scope.json
uv run --locked python benchmark/scale/run.py --diagnostic-file benchmark/scale/pickup-state.json --source-commit $version --markdown $book --output .local/benchmark/router-155-pickup.json
uv run --locked python benchmark/scale/smoke.py --expect-policy-scope --source-commit $version --markdown $book --output .local/benchmark/router-155-http.json
```

## 검증 범위

```powershell
$env:FAKE_LLM = 'true'
$env:LANGSMITH_TRACING = 'false'
uv run --locked pytest -q
```

`c7052de`에서 테스트 92개가 통과했습니다. 155조는 실제 Chroma·BM25를 사용한 FastAPI `TestClient`와 로컬 Uvicorn으로 JSON·SSE·수정·재조회·보류를 검증했습니다. 측정의 OpenAI 요청은 0회입니다. 실제 모델의 문장 품질, 새로운 사용자 질문의 성능, 운영 지연은 별도 평가가 필요합니다. [HTTP 점검 원본](benchmark/scale/results/155-c7052de-localhost.json)

가짜 모델 로컬 서버를 8000번 포트에 띄운 뒤 `uv run --locked python benchmark/benchmark.py`로 동기·비동기 경로의 지연과 처리량을 비교할 수 있습니다. 두 경로의 작업량은 다릅니다. [리팩토링 전 README와 실제 OpenAI 예시](https://github.com/kodonghui/ohgi-review-server/blob/d13fba9510a31f517b983f933e9348187b87741a/README.md)
