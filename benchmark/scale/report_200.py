"""Verify and publish the 200-page benchmarks from the archived receipts."""
import html
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from markdown_it import MarkdownIt

from app.core.config import PROJECT_ROOT
from benchmark.scale.eval_200 import GOLDEN, EMBED_PRICE, MIL, read_journal, usage_receipt
from benchmark.scale.index_200 import ARTIFACTS, digest, load_bundle, write_json


ENGINES = {"chroma": "노트북 Chroma", "pgvector": "OVH pgvector"}
PROVIDERS = {"gpt": "GPT", "claude": "Claude", "deepseek": "DeepSeek"}
REPORT = PROJECT_ROOT / "benchmark" / "scale" / "RESULTS_200.md"
TIT = "벡터 검색 k와 전체 답글 API 평가 · 오지 200페이지 Chroma·pgvector 비교"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dollar(value) -> str:
    return f"{Decimal(value):.8f}"


def table(headers: list[str], rows: list[list]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |" for row in rows],
    ])


def dataset_url(run: dict) -> str:
    url, separator, _ = run["url"].partition("/compare?")
    if not separator or "/datasets/" not in url:
        raise ValueError("결과에 LangSmith 데이터셋 비교 URL이 없습니다.")
    return url


def collect():
    bundle = load_bundle()
    golden_sha = digest(GOLDEN)
    golden = read(GOLDEN)
    dataset_ids = {case["id"] for case in golden["cases"]}
    lesson, api = {}, {}
    datasets = set()
    for engine in ENGINES:
        for label in ("k1", "k3", *PROVIDERS):
            name = f"{golden['version']}-{engine}-{label}-{golden_sha[:8]}"
            directory = ARTIFACTS / engine
            run = read(directory / f"{name}.json")
            config = run["configuration"]
            assert config["policy_sha256"] == bundle.manifest["corpus_sha256"]
            assert config["vectors_sha256"] == bundle.manifest["vector_sha256"]
            assert config["golden_sha256"] == golden_sha
            assert config["embedding_dimensions"] == 3072
            assert config["max_concurrency"] == 2
            datasets.add(dataset_url(run))
            assert run["summary"]["requests"] == len(run["rows"])
            repeats = 1 if label.startswith("k") else 3
            assert Counter(row["case_id"] for row in run["rows"]) == Counter({case: repeats for case in dataset_ids})
            assert len({row["run_id"] for row in run["rows"]}) == len(run["rows"])
            run["cost"] = usage_receipt(
                directory / f"{name}.usage.jsonl", directory / f"{name}.embeddings.jsonl", config["model"],
            )
            (lesson if label.startswith("k") else api)[(engine, label)] = run
    for label in ("k1", "k3", *PROVIDERS):
        candidates = lesson if label.startswith("k") else api
        left, right = (candidates[(engine, label)]["configuration"] for engine in ENGINES)
        exclude = {"engine", "host_role", "os", "algorithm", "experiment"}
        assert {a: b for a, b in left.items() if a not in exclude} == {a: b for a, b in right.items() if a not in exclude}
    if len(datasets) != 1:
        raise ValueError("기준 실행의 LangSmith 데이터셋이 일치하지 않습니다.")
    return bundle, golden, lesson, api


def interruption_cost() -> list[dict]:
    staging = ARTIFACTS / "pgvector"
    items = []
    for usage in sorted(staging.glob("*-c1bce3b0.usage.jsonl")):
        prefix = usage.name.split(".usage.jsonl")[0]
        rows = read_journal(staging / f"{prefix}.rows.jsonl")
        receipt = usage_receipt(usage, staging / f"{prefix}.embeddings.jsonl", "gpt-6-luna")
        items.append({"name": prefix, "saved_rows": len(rows), "cost": receipt})
    return items


def case_trace(run, row):
    tenant = dataset_url(run).split("/datasets/", 1)[0]
    return f"{tenant}/projects/p/{run['project_id']}/r/{row['run_id']}?poll=true"


def same_ordered_results(lesson, variant):
    a, b = (
        {row["case_id"]: row.get("sources") for row in lesson[engine, variant]["rows"]}
        for engine in ENGINES
    )
    return sum(a[case] is not None and a[case] == b[case] for case in a)


def render():
    bundle, golden, lesson, api = collect()
    dataset = dataset_url(next(iter(lesson.values())))
    corpus = read(ARTIFACTS / "corpus.json")
    interrupted = interruption_cost()
    build = Decimal(bundle.manifest["build_embedding_usage"]["tokens"]) * EMBED_PRICE / MIL
    course_cost = sum(Decimal(run["cost"]["measured_usd"]) for run in lesson.values())
    api_cost = sum(Decimal(run["cost"]["measured_usd"]) for run in api.values())
    extra_cost = sum(Decimal(run["cost"]["measured_usd"]) for run in interrupted)
    total = build + course_cost + api_cost + extra_cost
    all_api = sum(run["summary"]["requests"] for run in api.values())
    all_passed = sum(run["summary"]["passed"] for run in api.values())
    all_errors = sum(run["summary"]["http_errors"] for run in api.values())
    sentences = [
        "# " + TIT,
        "",
        f"① **문서와 골든셋.** [실행] {corpus['pages']}페이지에 제1~{corpus['articles']}조를 넣었습니다. "
        f"제1~155조의 원문 해시를 보존했고, 새 골든셋 {len(golden['cases'])}문항을 두 환경에서 공유했습니다.",
        "",
        "② **교재의 k 비교.** [실행] Chroma 적중은 "
        f"{lesson['chroma', 'k1']['summary']['retrieval_hit']}/{lesson['chroma', 'k1']['summary']['retrieval_eligible']}에서 "
        f"{lesson['chroma', 'k3']['summary']['retrieval_hit']}/{lesson['chroma', 'k3']['summary']['retrieval_eligible']}로 달라졌고, "
        f"pgvector는 {lesson['pgvector', 'k1']['summary']['retrieval_hit']}/{lesson['pgvector', 'k1']['summary']['retrieval_eligible']}에서 "
        f"{lesson['pgvector', 'k3']['summary']['retrieval_hit']}/{lesson['pgvector', 'k3']['summary']['retrieval_eligible']}로 달라졌습니다.",
        "",
        f"③ **전체 답글 API.** [실행] {all_api}회에서 {all_passed}회 통과했고 HTTP 오류는 {all_errors}회였습니다. "
        f"측정된 사용량에 단가를 적용한 합계는 **$\u007bollar\u007d**입니다.".replace(
            "$\u007bollar\u007d", "$" + dollar(total)),
        "",
        "## 질문과 출처",
        "",
        "교재 17번의 검색 개수 k를 바꾸면 정답 조문과 답변 정확도는 어떻게 달라지나요? "
        "같은 규정집과 골든셋을 노트북 Chroma와 OVH pgvector에 넣으면 실제 답글 API의 통과율과 비용은 어떻게 달라지나요?",
        "",
        "## ① 200페이지 규정집과 실행 환경",
        "",
        "[실행] 기존 155조 뒤에 가상 조문 450개를 추가했습니다. 각 조문 제목마다 하나의 문서로 나눴습니다. "
        "문서 제목·본문을 OpenAI API에 보내 3,072차원 벡터를 만들고, 같은 벡터 파일을 두 검색 저장소에 넣었습니다. "
        "평가 문항과 참고 답변은 조문을 바탕으로 작성한 합성 데이터입니다.",
        "",
        table(["검증 항목", "값"], [
            ["기존 155조 원문 SHA-256", f"`{corpus['original_155_sha256']}`"],
            ["200페이지 Markdown SHA-256", f"`{corpus['corpus_sha256']}`"],
            ["PDF SHA-256", f"`{corpus['pdf_sha256']}`"],
            ["공통 문서 벡터 SHA-256", f"`{bundle.manifest['vector_sha256']}`"],
            ["골든셋 SHA-256", f"`{digest(GOLDEN)}`"],
            ["실행 코드", f"`{api['chroma', 'gpt']['configuration']['code_commit']}`"],
        ]),
        "",
        "## ② 교재 17번의 k=1·3",
        "",
        "[사실] k는 첫 검색에서 꺼내는 조문의 개수입니다. **검색 적중**은 채점에 필요한 조문이 전부 포함됐을 때 1입니다. "
        "정답 조문이 없는 8문항은 적중 점수의 분모에서 제외합니다. 답변 정확도는 32문항 모두에서 같은 GPT 답변 모델과 "
        "같은 GPT 평가 모델로 체크했습니다. 복합 사례 8개는 최소 두 조문을 요구합니다.",
        "",
        table(["저장소", "k", "정답 조문 적중", "답변 정확", "트레이스 오류", "검색 p50 / p95 ms", "측정 USD"], [
            [ENGINES[engine], f"[{run['configuration']['k']}]({run['url']})",
             f"{s['retrieval_hit']}/{s['retrieval_eligible']}", f"{s['correct']}/{s['requests']}",
             s["errors"], f"{s['vector_ms_p50']} / {s['vector_ms_p95']}", dollar(run["cost"]["measured_usd"])]
            for (engine, _), run in lesson.items() for s in [run["summary"]]
        ]),
        "",
        table(["실험", "기존 조문 적중", "신규 조문 적중", "복합 조문 적중", "복합 답변 정확"], [
            [f"{ENGINES[engine]} {label}",
             *[f"{sum(bool(row['retrieval_hit']) for row in run['rows'] if row['case_id'].startswith(letter))}/8"
               for letter in "ONM"],
             f"{run['summary']['groups']['mixed']['passed']}/8"]
            for (engine, label), run in lesson.items()
        ]),
        "",
        f"[실행] 두 저장소의 정렬된 조문 목록이 일치한 수는 k=1에서 {same_ordered_results(lesson, 'k1')}/32, "
        f"k=3에서 {same_ordered_results(lesson, 'k3')}/32였습니다. 모델 답변은 각 환경에서 새로 생성했습니다.",
        "",
        "## ③ 실제 답글 API 반복",
        "",
        "[실행] Chroma는 Windows 노트북의 로컬 HNSW 코사인 검색, pgvector는 OVH Linux의 정확 코사인 검색입니다. "
        "두 환경에서 FastAPI TestClient로 실제 POST 답글 API를 호출하고, 다시 GET으로 저장된 텍스트를 읽었습니다. "
        "이 단계에서는 교재의 k 대신 벡터와 BM25 후보를 각각 최대 8개 가져오는 기존 API 경로를 같은 설정으로 실행했습니다.",
        "",
        "| 통과 조건 | 상태 일치 · 필수 조문 포함 · 필수 안내 문구 포함 · 저장 결과 일치 |",
        "| --- | --- |",
        "| 오승인 | 직원 확인이 필요한 문항에 API가 `ready`를 반환한 경우 |",
        "| 3회 통과 | 동일 문항의 세 반복이 모두 통과한 문항 수 |",
        "",
        table(["저장소", "모델", "통과/96", "3회 통과/32", "오승인", "HTTP 오류", "답글 p50/p95 초", "검색 p50/p95 ms"], [
            [ENGINES[engine], f"[{PROVIDERS[model]}]({run['url']})", s["passed"], s["all_repetitions_passed"],
             s["incorrect_auto_approvals"], s["http_errors"],
             f"{s['median_reply_ms']/1000:.3f}/{s['p95_reply_ms']/1000:.3f}",
             f"{s['vector_ms_p50']}/{s['vector_ms_p95']}"]
            for (engine, model), run in api.items() for s in [run["summary"]]
        ]),
        "",
        table(["저장소·모델", "기존/24", "신규/24", "복합/24", "직원 확인/21", "일반 후기/3"], [
            [f"{ENGINES[engine]} {PROVIDERS[model]}",
             *[s["groups"][group]["passed"] for group in ("existing", "new", "mixed", "staff", "general")]]
            for (engine, model), run in api.items() for s in [run["summary"]]
        ]),
        "",
        "검색 지연에는 검색 저장소 호출과 결과 조회가, 답글 지연에는 모델 요청과 API 처리가 포함됩니다. "
        "두 컴퓨터와 모델 네트워크가 달라 답글 지연 차이에는 이 환경 차이가 함께 들어갑니다.",
        "",
        "### 문항과 실제 결과",
        "",
        "[실행] M01은 공휴일이 아닌 금요일 오후 4시 매장 식사와 전화 포장을 묻습니다. "
        "기대값은 제1조와 제6조, 매장 식사 불가와 앱·홈페이지 포장 안내입니다. "
        "H01은 특정 고객의 적립 완료를 묻고, 기대 상태는 직원 확인입니다.",
        "",
        "[실행] 교재 k=3에서 M01의 Chroma 검색 순서는 "
        + ", ".join(next(row for row in lesson["chroma", "k3"]["rows"] if row["case_id"] == "M01").get("sources", []))
        + "입니다. 이 순서에서는 영업시간을 확인할 수 있고 전화 포장에 필요한 별도 조문은 빠졌습니다. "
        "전체 API의 조문 최종 선택과 저장 결과는 아래 요청별 트레이스에서 확인합니다.",
        "",
        table(["저장소·모델", "M01 실제 상태·인용", "H01 실제 상태"], [
            [f"{ENGINES[engine]} {PROVIDERS[model]}",
             " · ".join(f"[{row.get('status') or 'HTTP ' + str(row.get('http_status'))}"
                        f" {','.join(row.get('article_ids', []))}]({case_trace(run, row)})"
                        for row in run["rows"] if row["case_id"] == "M01"),
             " · ".join(f"[{row.get('status') or 'HTTP ' + str(row.get('http_status'))}]({case_trace(run, row)})"
                        for row in run["rows"] if row["case_id"] == "H01")]
            for (engine, model), run in api.items()
        ]),
        "",
        "## 비용과 실행 기록",
        "",
        table(["항목", "토큰/횟수", "측정 USD"], [
            ["공통 문서 임베딩", f"{bundle.manifest['build_embedding_usage']['tokens']}토큰 / {bundle.manifest['build_embedding_usage']['requests']}회", dollar(build)],
            ["두 환경 교재 k 비교 및 의미 채점", f"{sum(run['summary']['requests'] for run in lesson.values())}문항 실행", dollar(course_cost)],
            ["두 환경 전체 답글 API 및 질의 임베딩", f"{all_api}회", dollar(api_cost)],
            ["OVH 줄바꿈 정정 전 별도 호출", f"{sum(run['saved_rows'] for run in interrupted)}행 저장", dollar(extra_cost)],
            ["합계", "응답에 기록된 사용량 기준", dollar(total)],
        ]),
        "",
        table(["저장소·모델", "응답 사용량 수집 호출", "입력", "캐시 읽기", "캐시 쓰기", "출력", "질의 임베딩 토큰", "합계 USD"], [
            [f"{ENGINES[engine]} {PROVIDERS[model]}", c["model_usage"].get("calls_with_usage", 0),
             *[c["model_usage"].get(field, 0) for field in ("input", "cache_read", "cache_write", "output")],
             c["query_embedding_usage"]["tokens"], dollar(c["measured_usd"])]
            for (engine, model), run in api.items() for c in [run["cost"]]
        ]),
        "",
        "[사실] 2026-11-17에 확인한 표준 단가는 GPT-6 Luna와 Claude Haiku 5.5 모두 일반 입력 $0.10, "
        "캐시 읽기 $0.01, 단기 캐시 쓰기 $0.125, 출력 $0.50입니다. "
        "OpenAI large 임베딩은 입력 $0.13입니다. 단위는 각각 100만 토큰당 USD입니다. "
        "DeepSeek 릴레이 문서는 일일 무료 할당량을 명시합니다. "
        "토큰을 반환하지 않은 요청, OVH 기존 서버 이용료와 LangSmith 요금은 위 사용량 산정액에서 별도입니다. "
        "[GPT 단가](https://developers.openai.com/api/docs/models/gpt-6-luna) · "
        "[Claude 단가](https://platform.claude.com/docs/en/about-claude/pricing) · "
        "[임베딩 단가](https://developers.openai.com/api/docs/models/text-embedding-3-large) · "
        "[DeepSeek 릴레이 할당량](https://100milionseek.top/docs)",
        "",
        "[실행] Windows와 Git 추출본의 줄바꿈이 달라 최초 OVH 교재 실험 일부가 별도 데이터셋에 들어갔습니다. "
        "공통 골든셋 파일을 그대로 복사하고 해시를 확인한 뒤 같은 데이터셋으로 다시 실행했습니다. "
        "별도 실험은 위 점수 표와 분리했고, 기록된 사용량은 비용에 포함했습니다.",
        "",
        "## 재현 자료와 한계",
        "",
        "- [공통 LangSmith 데이터셋](" + dataset + ")의 문항 ID로 검색과 답글 트레이스를 대조합니다.",
        "- 200페이지 생성기는 `benchmark/scale/render_200.py`에, 두 검색 저장소는 `benchmark/scale/index_200.py`에, "
        "실행과 채점은 `benchmark/scale/eval_200.py`에 있습니다. 답글 API 데이터는 `data/langsmith_reply_golden_200_v1.json`입니다.",
        "- 로컬 `.local/ohgi-200p/`에는 PDF·Markdown·문서 벡터와 두 환경에서 회수한 응답 JSON 및 토큰 사용량 JSONL이 있습니다.",
        "- 복합 문항의 정확도는 GPT 자기 계열 평가 결과이며, 32개 합성 문항에서 얻은 결과입니다. "
        "실제 운영 규모와 장기 지연 분포는 별도 문제입니다.",
        "",
        "## 확인 질문",
        "",
        "필수 조문이 제1조와 제6조인 질문에서 k=1의 조문 적중이 어려운 이유를, 검색 개수와 채점 조건을 사용해 설명해 보세요.",
        "",
    ]
    cleanup = ARTIFACTS / "ovh-cleanup.json"
    if cleanup.is_file():
        proof = read(cleanup)
        assert proof["directory_absent"] and proof["postgres_stopped"]
        sentences.extend(["## OVH 정리 검증", "",
                          f"[실행] {proof['checked_at_utc']}에 결과 회수 해시를 확인한 후 실험 PostgreSQL을 종료했습니다. "
                          "실험 전용 DB·임시 실행 파일·로그가 있던 디렉터리의 부재를 SSH로 확인했습니다.", ""])
    markdown = "\n".join(sentences)
    REPORT.write_text(markdown, encoding="utf-8")
    write_study_html(markdown)
    compact = {
        "lesson": [dict(engine=engine, variant=label, project_id=run["project_id"],
                        url=run["url"], summary=run["summary"], cost=run["cost"])
                   for (engine, label), run in lesson.items()],
        "api": [dict(engine=engine, provider=label, project_id=run["project_id"],
                     url=run["url"], summary=run["summary"], cost=run["cost"])
                for (engine, label), run in api.items()],
        "costs": {"document": dollar(build), "lesson": dollar(course_cost), "api": dollar(api_cost),
                  "interruption": dollar(extra_cost), "total": dollar(total)},
        "requests": all_api, "passed": all_passed, "http_errors": all_errors,
        "interruptions": interrupted,
    }
    write_json(ARTIFACTS / "summary.json", compact)
    print(json.dumps({k: compact[k] for k in ("costs", "requests", "passed", "http_errors")}, ensure_ascii=False))


def write_study_html(markdown: str) -> None:
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    content = renderer.render(markdown)
    content = content.replace("<table>", '<div class="table-scroll"><table>').replace("</table>", "</table></div>")
    title = html.escape(TIT)
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>
:root {{color-scheme:light dark;--paper:#f7f5f1;--ink:#202820;--accent:#215c4e;--line:#cbcfc7;--panel:#eeeee7}}
@media(prefers-color-scheme:dark) {{:root {{--paper:#171c1b;--ink:#ebeee7;--accent:#a2dcc3;--line:#57625d;--panel:#242b28}}}}
* {{box-sizing:border-box}}
body {{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,sans-serif;word-break:keep-all;overflow-wrap:anywhere}}
main {{max-width:70ch;margin:auto;padding:clamp(1rem,4vw,3rem)}}
h1,h2,h3 {{line-height:1.24;letter-spacing:-.02em}}
h1 {{font-size:clamp(1.9rem,5vw,2.75rem);margin-top:1.4em}}
h2 {{margin-top:2.1em}}
p {{margin:1.1em 0}}
a {{color:var(--accent)}}
.table-scroll {{max-width:100%;overflow:auto;margin:1.3rem 0;border:1px solid var(--line);border-radius:.5rem}}
table {{width:100%;border-collapse:collapse;font-size:.9rem}}
th,td {{text-align:left;padding:.6em .78em;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}}
th {{background:var(--panel);min-width:4em}}
td code {{word-break:break-all}}
</style></head><body><main>{content}</main></body></html>"""
    (ARTIFACTS / "study-200p.html").write_text(document, encoding="utf-8")


if __name__ == "__main__":
    render()
