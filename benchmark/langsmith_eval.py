"""골든셋을 검증하고 FastAPI 답글 엔드포인트를 LangSmith에서 평가한다."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from time import perf_counter
from uuid import uuid5

import langsmith as ls
from fastapi import Request
from fastapi.testclient import TestClient
from langchain_core.callbacks import UsageMetadataCallbackHandler

from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app
from app.schemas.review import ReviewCreateRequest
from app.services.policy_index import REGULATIONS, load_articles
from app.services.reply_workflow import MAX_DRAFT_ATTEMPTS
from benchmark.langsmith_embeddings import EvaluationIndex, RESULTS_DIR
from benchmark.langsmith_providers import Provider, model_factory, provider_config, required_setting, setting


GOLDEN = PROJECT_ROOT / "data" / "langsmith_reply_golden_v1.json"


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_golden() -> dict:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = golden["cases"]
    articles = {article.id for article in load_articles()}
    if len(cases) != 12 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("골든셋에는 ID가 고유한 사례 12건이 필요합니다.")
    if sum(case["source"] == "synthetic" for case in cases) != 4:
        raise ValueError("골든셋에는 신규 합성 사례가 4건 필요합니다.")
    for case in cases:
        ReviewCreateRequest.model_validate(case["review"])
        expected = case["expected"]
        if expected["status"] not in {"ready", "needs_review"} or set(expected["article_ids"]) - articles:
            raise ValueError(f"{case['id']}의 기대 상태 또는 조문을 확인하세요.")
        for notice in expected["notices"]:
            re.compile(notice["pattern"])
    return golden


def langsmith_client() -> ls.Client:
    return ls.Client(
        api_key=required_setting("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        api_url=setting("LANGSMITH_ENDPOINT"),
        workspace_id=setting("LANGSMITH_WORKSPACE_ID"),
    )


def sync_dataset(client: ls.Client, golden: dict):
    name = f"{golden['version']}-{fingerprint(GOLDEN)[:12]}"
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description="오지 리뷰 답글 API 골든셋. 합성 및 강의 사례 12건.")
    examples = {
        uuid5(dataset.id, case["id"]): {
            "inputs": {"case_id": case["id"], "review": case["review"]},
            "outputs": {"expected": case["expected"]},
            "metadata": {"group": case["group"], "source": case["source"]},
        }
        for case in golden["cases"]
    }
    existing = {example.id: example for example in client.list_examples(dataset_id=dataset.id)}
    if set(existing) - set(examples) or any(
        example.inputs != examples[example_id]["inputs"]
        or example.outputs != examples[example_id]["outputs"]
        for example_id, example in existing.items()
    ):
        raise ValueError(f"LangSmith 데이터셋 {name}의 예제가 로컬 골든셋과 다릅니다.")
    missing = [
        {"id": example_id, **example}
        for example_id, example in examples.items()
        if example_id not in existing
    ]
    if missing:
        client.create_examples(dataset_id=dataset.id, examples=missing)
    return dataset


def invoke_reply(http: TestClient, inputs: dict) -> dict:
    tree = ls.get_current_run_tree()
    headers = tree.to_headers() if tree is not None else {}
    registered = http.post("/api/reviews", json=inputs["review"])
    registered.raise_for_status()
    review_path = f"/api/reviews/{registered.json()['reviewId']}"
    start = perf_counter()
    response = http.post(f"{review_path}/reply", headers=headers)
    elapsed_ms = round((perf_counter() - start) * 1000, 1)
    result = {"case_id": inputs["case_id"], "http_status": response.status_code, "reply_ms": elapsed_ms}
    if response.status_code != 200:
        content_type = response.headers.get("content-type", "")
        error = response.json() if "application/json" in content_type else {}
        return {**result, "error_code": error.get("code"), "error_detail": error.get("detail")}

    answer = response.json()
    stored = http.get(review_path)
    return {
        **result,
        "status": answer["status"],
        "article_ids": answer["articleIds"],
        "reply": answer["reply"],
        "attempts": answer["attempts"],
        "model": answer["model"],
        "stored_matches": stored.status_code == 200 and stored.json()["reply"] == answer["reply"],
    }


def grade(outputs: dict, expected: dict) -> dict:
    success = outputs.get("http_status") == 200
    status_match = success and outputs.get("status") == expected["status"]
    expected_ids = set(expected["article_ids"])
    actual_ids = set(outputs.get("article_ids", []))
    articles_match = success and (
        expected_ids <= actual_ids if expected_ids else not actual_ids
    )
    reply = outputs.get("reply") or ""
    missing_notices = [
        notice["description"] for notice in expected["notices"]
        if not re.search(notice["pattern"], reply, flags=re.IGNORECASE)
    ]
    notices_match = success and not missing_notices
    stored_matches = success and bool(outputs.get("stored_matches"))
    return {
        "passed": status_match and articles_match and notices_match and stored_matches,
        "status_match": status_match,
        "articles_match": articles_match,
        "notices_match": notices_match,
        "stored_matches": stored_matches,
        "incorrect_auto_approval": expected["status"] == "needs_review" and outputs.get("status") == "ready",
        "missing_notices": missing_notices,
    }


def assess_reply(outputs: dict, reference_outputs: dict) -> dict:
    marks = grade(outputs or {}, reference_outputs["expected"])
    feedback = [
        {"key": metric, "score": int(value)}
        for metric, value in marks.items()
        if isinstance(value, bool)
    ]
    if marks["missing_notices"]:
        next(row for row in feedback if row["key"] == "notices_match")["comment"] = "; ".join(marks["missing_notices"])
    return {"results": feedback}


def fake_smoke(golden: dict, index: EvaluationIndex) -> dict:
    settings = Settings(
        _env_file=None, openai_api_key=None, fake_llm=True,
        fake_llm_delay_seconds=0, runtime_dir=index.runtime_dir,
    )
    case = next(item for item in golden["cases"] if item["id"] == "M01")
    with ls.tracing_context(enabled=False), TestClient(
        create_app(settings, policy_index_loader=index.load),
    ) as http:
        outputs = invoke_reply(http, {"case_id": case["id"], "review": case["review"]})
    if outputs.get("status") != "ready" or outputs.get("model") != "fake-llm" or not outputs.get("stored_matches"):
        raise RuntimeError("가짜 모델로 답글 API와 정책 인덱스를 확인하지 못했습니다.")
    return outputs


def summarize(rows: list[dict], repetitions: int) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    latencies = [row["reply_ms"] for row in rows if row.get("reply_ms") is not None]
    return {
        "requests": len(rows),
        "matched_status": sum(row["status_match"] for row in rows),
        "matched_articles": sum(row["articles_match"] for row in rows),
        "matched_notices": sum(row["notices_match"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "all_repetitions_passed": sum(
            len(items) == repetitions and all(row["passed"] for row in items)
            for items in grouped.values()
        ),
        "incorrect_auto_approvals": sum(row["incorrect_auto_approval"] for row in rows),
        "http_errors": sum(row.get("http_status") != 200 for row in rows),
        "median_reply_ms": median(latencies) if latencies else None,
        "max_reply_ms": max(latencies, default=None),
        "runs_per_case": dict(sorted(Counter(row["case_id"] for row in rows).items())),
    }


def run_experiment(
    client: ls.Client, dataset, golden: dict, index: EvaluationIndex,
    provider: Provider, repetitions: int = 3, concurrency: int = 3,
) -> dict:
    config = provider_config(provider)
    usage = UsageMetadataCallbackHandler()
    is_relay = provider == "deepseek"
    model_timeout = 180 if is_relay else 60
    # Reserve time for each draft and review, plus local work.
    reply_timeout = 2 * MAX_DRAFT_ATTEMPTS * model_timeout + 30 if is_relay else 180
    settings = Settings(
        _env_file=None,
        fake_llm=False,
        runtime_dir=index.runtime_dir,
        openai_model=config.model,
        openai_api_key=config.api_key if provider == "gpt" else None,
        openai_reasoning_effort="none",
        # The relay uses some of its output budget for reasoning before returning JSON.
        openai_max_output_tokens=1800 if is_relay else 500,
        openai_timeout_seconds=model_timeout,
        reply_timeout_seconds=reply_timeout,
    )
    run_metadata = {
        "provider": provider,
        "model": config.model,
        "policy_sha256": fingerprint(REGULATIONS),
        "golden_sha256": fingerprint(GOLDEN),
        "embedding_model": index.embedding.model,
        "embedding_dimensions": index.embedding.dimensions,
        "model_timeout_seconds": model_timeout,
        "reply_timeout_seconds": reply_timeout,
        "max_output_tokens": settings.openai_max_output_tokens,
        "native_thinking": "relay default" if is_relay else "disabled",
        "transport_max_retries": 0 if is_relay else 1,
        "provider_prompt_cache": {"gpt": "prefix key", "claude": "ephemeral", "deepseek": "relay"}[provider],
        "num_repetitions": repetitions,
        "max_concurrency": concurrency,
    }
    cache_key = f"ohgi-reply-{fingerprint(GOLDEN)[:12]}"
    application = create_app(
        settings, model_factory(config, usage, cache_key), policy_index_loader=index.load,
    )

    @application.middleware("http")
    async def link_trace(request: Request, call_next):
        if "langsmith-trace" in request.headers:
            parent = ls.RunTree.from_headers(request.headers, client=client)
            with ls.tracing_context(parent=parent, client=client, enabled=True):
                return await call_next(request)
        return await call_next(request)

    with TestClient(application, raise_server_exceptions=False) as http:
        with ls.tracing_context(enabled=False):
            if http.get("/health").json()["model"] != config.model:
                raise RuntimeError(f"{provider} 모델 설정과 서버 모델이 다릅니다.")
            # A relay can spend its full timeout before the experiment is visible.
            if not is_relay:
                smoke = next(case for case in golden["cases"] if case["id"] == "G03")
                probe = invoke_reply(http, {"case_id": smoke["id"], "review": smoke["review"]})
                if probe["http_status"] != 200 or not probe.get("attempts"):
                    detail = probe.get("error_detail") or f"HTTP {probe['http_status']}"
                    raise RuntimeError(f"{provider}의 실제 모델 확인 실패. {detail}")
        smoke_usage = dict(usage.usage_metadata)
        usage.usage_metadata.clear()
        print(f"{provider} {config.model} LangSmith 반복 실험을 시작합니다.", flush=True)

        def target(inputs: dict) -> dict:
            return invoke_reply(http, inputs)

        embedding_before = index.usage.snapshot()
        prefix = f"{golden['version']}-{provider}" + ("-180s" if is_relay else "") + "-oaiemb"
        experiment = ls.evaluate(
            target, data=dataset.name, evaluators=[assess_reply], client=client,
            experiment_prefix=prefix,
            num_repetitions=repetitions, max_concurrency=concurrency,
            metadata=run_metadata,
        )
        rows = []
        for result in experiment:
            run, example = result["run"], result["example"]
            output = run.outputs or {}
            rows.append({
                **output,
                "case_id": example.inputs["case_id"],
                **grade(output, example.outputs["expected"]),
                "run_id": str(run.id),
                "trace_error": bool(run.error),
            })
        client.flush(timeout=30)
        embedding_after = index.usage.snapshot()

    report = {
        "provider": provider,
        "model": config.model,
        "experiment_name": experiment.experiment_name,
        "url": experiment.url,
        "dataset_url": dataset.url,
        "configuration": run_metadata,
        "embedding_usage": {
            "index_build_in_session": index.build_usage,
            "experiment_queries": {key: embedding_after[key] - value for key, value in embedding_before.items()},
        },
        "summary": summarize(rows, repetitions),
        "usage": usage.usage_metadata,
        "smoke_usage": smoke_usage,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"{experiment.experiment_name}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
