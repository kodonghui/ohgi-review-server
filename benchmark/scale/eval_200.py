"""Run matched 200-page lesson and full reply API evaluations."""
import argparse
import json
import platform
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import UUID

import langsmith as ls
import numpy as np
from fastapi import Request
from fastapi.testclient import TestClient
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.outputs import ChatGeneration
from langchain_openai import ChatOpenAI
from langsmith.utils import LangSmithNotFoundError
from pydantic import BaseModel

from app.core.config import PROJECT_ROOT, Settings
from app.main import create_app
from app.schemas.review import ReviewCreateRequest
from app.services.reply_workflow import MAX_DRAFT_ATTEMPTS
from benchmark.langsmith_eval import assess_reply, grade, invoke_reply, langsmith_client, summarize, sync_dataset
from benchmark.langsmith_providers import PROVIDERS, model_factory, provider_config, setting
from benchmark.scale.index_200 import (
    ARTIFACTS, BOOK, MeasuredIndex, build_bundle, digest, load_bundle, metered_embeddings, open_store, write_json,
)


GOLDEN = PROJECT_ROOT / "data" / "langsmith_reply_golden_200_v1.json"
CONCURRENCY = 2
REPETITIONS = 3
PILOT = {"O01", "N01", "M01", "H01", "H08"}
MIL = Decimal(1_000_000)
PRICING = {
    "gpt-6-luna": ("0.10", "0.01", "0.125", "0.50"),
    "claude-haiku-5-5": ("0.10", "0.01", "0.125", "0.50"),
    "deepseek-v4.1-flash-free": ("0", "0", "0", "0"),
}
EMBED_PRICE = Decimal("0.13")


class UsageLedger(UsageMetadataCallbackHandler):
    def __init__(self, path: Path, fallback_model: str):
        super().__init__()
        self.path = path
        self.fallback_model = fallback_model
        self._journal_lock = Lock()

    def on_llm_end(self, response, *, run_id, **kwargs):
        super().on_llm_end(response, run_id=run_id, **kwargs)
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        if not isinstance(generation, ChatGeneration):
            return
        message = generation.message
        usage = message.usage_metadata
        if usage is None:
            return
        self._append({
            "run_id": str(run_id), "state": "response",
            "observed_model": message.response_metadata.get("model_name") or message.response_metadata.get("model"),
            "pricing_model": self.fallback_model, "usage": usage,
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._append({
            "run_id": str(run_id), "state": "error", "pricing_model": self.fallback_model,
            "exception": type(error).__name__, "http_status": getattr(error, "status_code", None),
        })

    def _append(self, row):
        row["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
        with self._journal_lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def usage_receipt(path: Path, embedding_path: Path, model: str) -> dict:
    queries = read_journal(embedding_path)
    query_usage = {"requests": len(queries), "tokens": sum(row["tokens"] for row in queries)}
    seen = {row["run_id"]: row for row in read_journal(path)}
    ledgers = list(seen.values())
    tokens = Counter()
    errors = Counter()
    for row in ledgers:
        if row["state"] == "error":
            errors[f"{row['exception']}:{row['http_status']}"] += 1
            continue
        usage = row["usage"]
        details = usage.get("input_token_details", {})
        read = details.get("cache_read", 0)
        long_write = details.get("ephemeral_1h_input_tokens", 0)
        write = max(details.get("cache_creation", 0), details.get("ephemeral_5m_input_tokens", 0) + long_write)
        tokens["input"] += usage["input_tokens"]
        tokens["cache_read"] += read
        tokens["cache_write"] += write
        tokens["cache_write_1h"] += long_write
        tokens["regular_input"] += usage["input_tokens"] - read - write
        tokens["output"] += usage["output_tokens"]
        tokens["total"] += usage["total_tokens"]
        tokens["calls_with_usage"] += 1
    regular, read, write, output = map(Decimal, PRICING[model])
    short_writes = tokens["cache_write"] - tokens["cache_write_1h"]
    long_write_rate = Decimal("0.20") if model == "claude-haiku-5-5" else write
    model_cost = (tokens["regular_input"] * regular + tokens["cache_read"] * read + short_writes * write
                  + tokens["cache_write_1h"] * long_write_rate + tokens["output"] * output) / MIL
    embedding_cost = Decimal(query_usage["tokens"]) * EMBED_PRICE / MIL
    return {
        "model_usage": dict(tokens), "provider_errors": dict(errors),
        "query_embedding_usage": query_usage,
        "model_usd": str(model_cost), "query_embedding_usd": str(embedding_cost),
        "measured_usd": str(model_cost + embedding_cost),
    }


def load_golden(bundle) -> dict:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = golden["cases"]
    ids = {case["id"] for case in cases}
    articles = set(bundle.manifest["article_ids"])
    if len(cases) != 32 or len(ids) != 32 or not PILOT <= ids:
        raise ValueError("32개의 고유한 문항과 사전 점검 문항을 확인하세요.")
    for case in cases:
        ReviewCreateRequest.model_validate(case["review"])
        expected = case["expected"]
        if expected["status"] not in {"ready", "needs_review"} or set(expected["article_ids"]) - articles:
            raise ValueError(f"기대 조문 또는 상태가 잘못됐습니다. {case['id']}")
        for notice in expected["notices"]:
            re.compile(notice["pattern"])
    return golden


def experiment_info(client, dataset, name: str, metadata: dict):
    try:
        project = client.read_project(project_name=name)
    except LangSmithNotFoundError:
        return client.create_project(name, reference_dataset_id=dataset.id, metadata=metadata)
    if project.reference_dataset_id != dataset.id:
        raise ValueError(f"실험 {name}의 데이터셋이 변경됐습니다.")
    return project


def experiment_name(golden: dict, backend: str, label: str, variant: str = "") -> str:
    base = f"{golden['version']}-{backend}-{label}-{digest(GOLDEN)[:8]}"
    return f"{base}-{variant}" if variant else base


def assess_lesson(outputs: dict, reference_outputs: dict) -> dict:
    outputs = outputs or {}
    expected = reference_outputs["expected"]["article_ids"]
    results = [{"key": "correctness", "score": int(outputs.get("correct", False)),
                "comment": outputs.get("judge_reason", "")}]
    if expected:
        hit = set(expected) <= set(outputs.get("sources", []))
        results.append({"key": "retrieval_hit", "score": int(hit)})
    return {"results": results}


def evaluate_rows(client, dataset, examples, target, evaluator, name, metadata, journal, repetitions, score_row):
    project = experiment_info(client, dataset, name, metadata)
    by_example = {example.id: example for example in examples}
    rows = read_journal(journal)
    counts = Counter(row["case_id"] for row in rows)
    known = {row["run_id"] for row in rows}

    def append(run, example):
        raw = run.outputs or {}
        row = {**raw, **score_row(raw, example.outputs["expected"]), "case_id": example.inputs["case_id"],
               "run_id": str(run.id), "trace_error": bool(run.error)}
        rows.append(row)
        counts[row["case_id"]] += 1
        known.add(row["run_id"])
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    if rows or getattr(project, "run_count", 0):
        for run in client.list_runs(project_id=project.id, is_root=True):
            example = by_example.get(run.reference_example_id)
            if (str(run.id) not in known and run.end_time is not None and example is not None
                    and counts[example.inputs["case_id"]] < repetitions):
                append(run, example)

    def evaluate(subset, repeats):
        result = ls.evaluate(
            target, data=subset, evaluators=[evaluator], client=client, experiment=project,
            max_concurrency=CONCURRENCY, num_repetitions=repeats, blocking=False,
        )
        for entry in result:
            append(entry["run"], entry["example"])
        client.flush(timeout=30)

    pilot = [example for example in examples
             if example.inputs["case_id"] in PILOT and not counts[example.inputs["case_id"]]]
    if pilot:
        evaluate(pilot, 1)
        errors = sum(row.get("http_status", 200) != 200 or row["trace_error"]
                     for row in rows if row["case_id"] in PILOT)
        print(json.dumps({
            "experiment": name, "stage": "pilot", "observed": len(pilot),
            "http_or_trace_errors": errors,
        }), flush=True)
        if errors == len(PILOT):
            raise RuntimeError(f"{name}의 사전 점검이 모두 오류입니다.")

    remaining = defaultdict(list)
    for example in examples:
        missing = repetitions - counts[example.inputs["case_id"]]
        if missing > 0:
            remaining[missing].append(example)
    for repeats, subset in sorted(remaining.items(), reverse=True):
        evaluate(subset, repeats)
    expected = Counter({example.inputs["case_id"]: repetitions for example in examples})
    if counts != expected:
        raise RuntimeError(f"{name}에 예상한 반복 결과가 모이지 않았습니다.")
    url = dataset.url.split("?")[0] + f"/compare?selectedSessions={project.id}"
    return rows, project.id, url


def lesson_grade(outputs: dict, expected: dict) -> dict:
    ids = set(expected["article_ids"])
    actual = set(outputs.get("sources", []))
    return {"retrieval_hit": ids <= actual if ids else None,
            "required_articles": len(ids), "found_required_articles": len(ids & actual),
            "correct": outputs.get("correct", False)}


def percentile(values: list[float], p: float) -> float | None:
    return round(float(np.percentile(values, p)), 3) if values else None


def grouped_scores(rows: list[dict], golden: dict, *, metric: str) -> dict:
    id_group = {case["id"]: case["group"] for case in golden["cases"]}
    groups = defaultdict(list)
    for row in rows:
        groups[id_group[row["case_id"]]].append(row)
    return {group: {"passed": sum(bool(row.get(metric)) for row in items), "requests": len(items)}
            for group, items in groups.items()}


class JudgeResult(BaseModel):
    correct: bool
    reason: str


def run_lesson(client, dataset, examples, golden, bundle, store, backend, k, directory, variant=""):
    config = provider_config("gpt")
    name = experiment_name(golden, backend, f"k{k}", variant)
    report_path = directory / f"{name}.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    usage_path = directory / f"{name}.usage.jsonl"
    embedding_path = directory / f"{name}.embeddings.jsonl"
    usage = UsageLedger(usage_path, config.model)
    chat = ChatOpenAI(
        model=config.model, api_key=config.api_key, base_url="https://api.openai.com/v1", timeout=60,
        max_retries=1, max_tokens=500, reasoning={"effort": "none"}, use_responses_api=True,
        prompt_cache_key=f"{golden['version']}-lesson", callbacks=[usage],
    )
    judge = chat.with_structured_output(JudgeResult, method="json_schema", strict=True)
    by_case = {case["id"]: case["expected"] for case in golden["cases"]}
    started = perf_counter()
    with metered_embeddings(embedding_path) as (embedding, _):
        index = MeasuredIndex(store, bundle.documents, embedding)

        @ls.traceable(name="course_answer", run_type="chain")
        def answer_question(question, documents):
            context = "\n\n".join(doc.page_content for doc in documents)
            reply = chat.invoke([
                ("system", "오지 레스토랑 규정에 근거해 한국어로 답하세요. 고객 개인의 실제 기록은 "
                 "확인할 수 없으면 직원 확인을 안내하세요. 감사 후기에는 짧게 응답하고 "
                 "공급된 규정에 없는 사실은 단정하지 마세요."),
                ("human", f"조문\n{context}\n\n고객 문장\n{question}"),
            ])
            return reply.text

        @ls.traceable(name="course_correctness", run_type="chain")
        def judge_answer(question, reference, actual):
            return judge.invoke([
                ("system", "모범답안의 필수 사실을 모두 담고 모순이 없으면 correct=true. "
                 "복합 질문은 모든 조건을 답해야 합니다. 실제 상태 문의는 상태를 단정하지 않고 "
                 "확인 필요를 안내해야 합니다. 어투 차이는 허용하세요."),
                ("human", f"질문\n{question}\n\n모범답안\n{reference}\n\n실제 답변\n{actual}"),
            ])

        def target(inputs):
            start = perf_counter()
            case_id = inputs["case_id"]
            question = inputs["review"]["content"]
            docs = index.query(question, k)
            answer = answer_question(question, docs)
            verdict = judge_answer(question, by_case[case_id]["rationale"], answer)
            return {"case_id": case_id, "sources": [doc.id for doc in docs], "answer": answer,
                    "correct": verdict.correct, "judge_reason": verdict.reason,
                    "elapsed_ms": round((perf_counter() - start) * 1000, 1)}

        metadata = common_metadata(bundle, backend, name, config.model, phase="lesson", k=k, repetitions=1)
        rows, project, url = evaluate_rows(client, dataset, examples, target, assess_lesson, name, metadata,
                                           directory / f"{name}.rows.jsonl", 1, lesson_grade)
        applicable = [row for row in rows if row["retrieval_hit"] is not None]
        report = {
            "configuration": metadata, "name": name, "project_id": str(project), "url": url,
            "summary": {
                "requests": len(rows), "retrieval_eligible": len(applicable),
                "retrieval_hit": sum(row["retrieval_hit"] for row in applicable),
                "correct": sum(bool(row["correct"]) for row in rows),
                "errors": sum(row["trace_error"] for row in rows),
                "vector_ms_p50": percentile(index.times(), 50), "vector_ms_p95": percentile(index.times(), 95),
                "elapsed_seconds": round(perf_counter() - started, 3),
                "groups": grouped_scores(rows, golden, metric="correct"),
            },
            "cost": usage_receipt(usage_path, embedding_path, config.model),
            "rows": rows,
        }
    write_json(report_path, report)
    print(name, json.dumps({"summary": report["summary"], "cost": report["cost"]}, ensure_ascii=False), flush=True)
    return report


def common_metadata(bundle, backend, name, model, *, phase, repetitions, **extra):
    return {
        "experiment": name, "engine": backend, "phase": phase, "model": model, "os": platform.system(),
        "host_role": "OVH Singapore" if backend == "pgvector" else "Windows laptop",
        "algorithm": "exact cosine" if backend == "pgvector" else "HNSW cosine",
        "policy_sha256": bundle.manifest["corpus_sha256"], "vectors_sha256": bundle.manifest["vector_sha256"],
        "golden_sha256": digest(GOLDEN), "embedding_model": bundle.manifest["embedding_model"],
        "embedding_dimensions": bundle.manifest["dimensions"], "code_commit": setting("OHGI_CODE_COMMIT"),
        "num_repetitions": repetitions, "max_concurrency": CONCURRENCY, **extra,
    }


def run_api(client, dataset, examples, golden, bundle, store, backend, provider, directory, variant=""):
    config = provider_config(provider)
    name = experiment_name(golden, backend, provider, variant)
    report_path = directory / f"{name}.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    usage_path = directory / f"{name}.usage.jsonl"
    embedding_path = directory / f"{name}.embeddings.jsonl"
    usage = UsageLedger(usage_path, config.model)
    relay = provider == "deepseek"
    model_timeout = 180 if relay else 60
    reply_timeout = 2 * MAX_DRAFT_ATTEMPTS * model_timeout + 30 if relay else 180
    settings = Settings(
        _env_file=None, fake_llm=False, runtime_dir=directory / "runtime" / provider,
        openai_model=config.model, openai_api_key=config.api_key if provider == "gpt" else None,
        openai_reasoning_effort="none", openai_max_output_tokens=1800 if relay else 500,
        openai_timeout_seconds=model_timeout, reply_timeout_seconds=reply_timeout,
    )
    started = perf_counter()
    with metered_embeddings(embedding_path) as (embedding, _):
        index = MeasuredIndex(store, bundle.documents, embedding)
        cache_key = f"{golden['version']}-{digest(GOLDEN)[:8]}"
        app = create_app(settings, model_factory(config, usage, cache_key),
                         policy_index_loader=lambda _: index, policy_source=BOOK)

        @app.middleware("http")
        async def link_trace(request: Request, call_next):
            if "langsmith-trace" in request.headers:
                parent = ls.RunTree.from_headers(request.headers, client=client)
                with ls.tracing_context(parent=parent, client=client, enabled=True):
                    return await call_next(request)
            return await call_next(request)

        metadata = common_metadata(
            bundle, backend, name, config.model, phase="api", repetitions=REPETITIONS,
            provider=provider, model_timeout_seconds=model_timeout, reply_timeout_seconds=reply_timeout,
            candidate_count=8, ranking="BM25 + vector, RRF", max_output_tokens=settings.openai_max_output_tokens,
            transport_max_retries=0 if relay else 1,
            provider_prompt_cache={"gpt": "prefix key", "claude": "ephemeral", "deepseek": "relay"}[provider],
        )
        with TestClient(app, raise_server_exceptions=False) as http:
            if http.get("/health").json()["model"] != config.model:
                raise RuntimeError(f"{provider} API에 연결된 모델 ID가 설정과 다릅니다.")

            def target(inputs):
                return invoke_reply(http, inputs)

            rows, project, url = evaluate_rows(client, dataset, examples, target, assess_reply, name, metadata,
                                               directory / f"{name}.rows.jsonl", REPETITIONS, grade)

        summary = {
            **summarize(rows, REPETITIONS),
            "p95_reply_ms": percentile([row["reply_ms"] for row in rows if "reply_ms" in row], 95),
            "vector_ms_p50": percentile(index.times(), 50), "vector_ms_p95": percentile(index.times(), 95),
            "vector_queries_timed": len(index.times()), "elapsed_seconds": round(perf_counter() - started, 3),
            "groups": grouped_scores(rows, golden, metric="passed"),
        }
        report = {
            "configuration": metadata, "name": name, "project_id": str(project), "url": url, "summary": summary,
            "cost": usage_receipt(usage_path, embedding_path, config.model), "rows": rows,
        }
    write_json(report_path, report)
    print(name, json.dumps({"summary": summary, "cost": report["cost"]}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["chroma", "pgvector"])
    parser.add_argument("stage", choices=["dataset", "lesson", "api"])
    parser.add_argument("--provider", choices=PROVIDERS)
    parser.add_argument("--variant", default="", help="Experiment suffix and separate result directory")
    parser.add_argument("--dataset-id", type=UUID, help="Reuse an existing LangSmith dataset without editing it")
    args = parser.parse_args()
    if args.variant and not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.variant):
        parser.error("--variant accepts lowercase letters, digits and hyphens")
    for provider in PROVIDERS:
        model = provider_config(provider).model
        if model not in PRICING:
            raise ValueError(f"비용표에 {model}의 단가가 없습니다.")
    bundle = build_bundle() if args.backend == "chroma" else load_bundle()
    golden = load_golden(bundle)
    client = langsmith_client()
    dataset = (client.read_dataset(dataset_id=args.dataset_id) if args.dataset_id
               else sync_dataset(client, golden, GOLDEN))
    examples = sorted(client.list_examples(dataset_id=dataset.id), key=lambda example: example.inputs["case_id"])
    expected = {case["id"]: (case["review"], {"expected": case["expected"]}) for case in golden["cases"]}
    observed = {case.inputs["case_id"]: (case.inputs["review"], case.outputs) for case in examples}
    if len(examples) != len(expected) or observed != expected:
        raise RuntimeError("LangSmith의 골든셋이 로컬 입력 및 채점 기준과 다릅니다.")
    print(json.dumps({"dataset": dataset.name, "dataset_id": str(dataset.id), "dataset_url": dataset.url,
                      "examples": len(examples), "golden_sha256": digest(GOLDEN)}, ensure_ascii=False), flush=True)
    if args.stage == "dataset":
        return

    directory = ARTIFACTS / args.variant / args.backend
    directory.mkdir(parents=True, exist_ok=True)
    with open_store(args.backend, bundle) as store:
        if args.stage == "lesson":
            for k in (1, 3):
                run_lesson(client, dataset, examples, golden, bundle, store, args.backend, k, directory, args.variant)
        elif args.provider is not None:
            run_api(client, dataset, examples, golden, bundle, store, args.backend, args.provider, directory, args.variant)
        else:
            with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as executor:
                jobs = [executor.submit(
                    run_api, langsmith_client(), dataset, examples, golden, bundle, store, args.backend, provider,
                    directory, args.variant,
                ) for provider in PROVIDERS]
                for job in jobs:
                    job.result(timeout=None)


if __name__ == "__main__":
    main()
