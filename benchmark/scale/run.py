"""Measure corpus growth on committed app code, with a separate synthetic query set."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = REPO / "data" / "ohgi_regulations.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def link_model_file(source, target):
    try:
        os.link(source, target)
        return target
    except OSError:
        return shutil.copy2(source, target)


def measure_version(source_commit, source_archive, temp, label, policy, output, runs, diagnostic_file):
    snapshot = temp / label
    with ZipFile(source_archive) as archive:
        archive.extractall(snapshot)
    policy_path = snapshot / "data" / "ohgi_regulations.md"
    if policy:
        shutil.copyfile(policy, policy_path)

    runtime = snapshot / ".local"
    shutil.copytree(REPO / ".local" / "embedding-models", runtime / "embedding-models", copy_function=link_model_file)
    env = {**os.environ, "PYTHONUTF8": "1", "FAKE_LLM": "true", "LANGSMITH_TRACING": "false", "HF_HUB_OFFLINE": "1"}
    env.pop("OPENAI_API_KEY", None)
    command = [sys.executable, str(__file__), "--worker", "--runs", str(runs), "--output", str(output)]
    if diagnostic_file:
        command.extend(("--diagnostic-file", str(diagnostic_file)))
    completed = subprocess.run(
        command,
        cwd=snapshot, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"{label} 측정 실패\n{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}")
    result = json.loads(output.read_text(encoding="utf-8"))
    result.update(label=label, source_commit=source_commit)
    return result


def all_ids(row, key):
    return set(row["required"]).issubset(row[key])


def summarize(rows):
    from statistics import median, quantiles

    answerable = [row for row in rows if row["required"]]
    staff = [row for row in rows if not row["required"] and row.get("expected", "needs_review") == "needs_review"]
    api_samples = [sample for row in rows for sample in row["api_samples_ms"]]
    search_samples = [sample for row in rows for sample in row["search_samples_ms"]]
    return {
        "requests": len(rows),
        "with_reference": len(answerable),
        "full_top3_vector": sum(all_ids(row, "vector_top3") for row in answerable),
        "full_top3_bm25": sum(all_ids(row, "bm25_top3") for row in answerable),
        "full_top3_rrf": sum(all_ids(row, "rrf_top3") for row in answerable),
        "full_candidate_union": sum(all_ids(row, "candidate_union") for row in answerable),
        "reference_spans_in_union": sum(row.get("source_spans_found", False) for row in answerable),
        "reference_spans_saved": sum(row.get("answer_spans_found", False) for row in answerable),
        "full_api_retrieval": sum(all_ids(row, "articles") for row in answerable),
        "full_api_save": sum(row["saved"] and all_ids(row, "answer_articles") for row in answerable),
        "full_save_with_spans": sum(
            row["saved"] and all_ids(row, "answer_articles") and row.get("answer_spans_found", False)
            for row in answerable
        ),
        "requires_staff": len(staff),
        "staff_correctly_held": sum(row["status"] == "needs_review" and not row["saved"] for row in staff),
        "saved_missing_reference": sum(row["saved"] and not all_ids(row, "answer_articles") for row in answerable),
        "retries": sum(bool(row["retried_questions"]) for row in rows),
        "api_p50_ms": round(median(api_samples), 3),
        "api_p95_ms": round(quantiles(api_samples, n=20, method="inclusive")[18], 3),
        "search_p50_ms": round(median(search_samples), 3),
        "search_p95_ms": round(quantiles(search_samples, n=20, method="inclusive")[18], 3),
    }


def normalize(text):
    return " ".join(text.split())


def answer_contains_span(span, reply):
    answer = normalize(reply)
    # The existing fixed notice has the same meaning in polite speech.
    if span == "주류는 배달하지 않는다":
        return re.search(r"주류는 배달하지 (?:않는다|않습니다)", answer) is not None
    return span in answer


async def observe_candidates(index, cases):
    from app.services.hybrid_retriever import HybridRetriever
    from app.services.policy_index import load_articles
    from app.services.policy_topics import questions_in

    retriever = HybridRetriever(index, load_articles())
    seen = {}
    for case in cases:
        rankings = await retriever.search(case["question"])
        union = rankings.hybrid
        topics, reason = questions_in(case["question"])
        source = {doc.id: normalize(doc.page_content) for doc in union}
        labeled_source = " ".join(source.get(article, "") for article in case["required"])
        seen[case["id"]] = {
            "vector_top3": [doc.id for doc in rankings.vector[:3]],
            "bm25_top3": [doc.id for doc in rankings.bm25[:3]],
            "rrf_top3": [doc.id for doc in rankings.hybrid[:3]],
            "candidate_union": [doc.id for doc in union],
            "source_spans_found": bool(case.get("spans")) and all(span in labeled_source for span in case["spans"]),
            "router_topics": [rule.topic for rule, _ in topics],
            "router_reason": reason,
        }
    return seen


def worker(output, runs, diagnostic_file):
    import asyncio
    from importlib.metadata import version
    from time import perf_counter
    from unittest.mock import patch

    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(Path.cwd()))
    from fastapi.testclient import TestClient
    from policy_quality import capture_search, load_cases, measure, review, server
    from app.core.config import Settings
    from app.services.policy_index import build_policy_index, open_policy_index

    cases = load_cases()
    for case in cases:
        if not case["required"]:
            case["expected"] = "ready" if case["id"] == "N01" else "needs_review"
        if case["id"] == "G15":
            case["expected"] = "needs_review"
    cases.extend(json.loads((HERE / "questions.json").read_text(encoding="utf-8")))
    if diagnostic_file:
        cases = json.loads(diagnostic_file.read_text(encoding="utf-8"))
    warmup_indices = (0, len(cases) // 4, len(cases) // 2) if diagnostic_file else (0, 13, 21)
    runtime = Path.cwd() / ".local"
    articles_by_id = {
        article.id: normalize(article.page_content)
        for article in server.load_articles()
    }
    for case in cases:
        if not case.get("spans") or any(article not in articles_by_id for article in case["required"]):
            continue
        source = " ".join(articles_by_id[article] for article in case["required"])
        assert all(span in source for span in case["spans"]), case["id"]
    settings = Settings(
        _env_file=None, fake_llm=True, fake_llm_delay_seconds=0, openai_api_key=None, runtime_dir=runtime,
    )

    start = perf_counter()
    manifest = build_policy_index(runtime)
    build_seconds = perf_counter() - start
    index_bytes = sum(
        path.stat().st_size for path in (runtime / "policy-indexes" / manifest["directory"]).rglob("*")
        if path.is_file()
    )
    active, observe = capture_search()
    with patch.object(server, "get_settings", lambda: settings), observe:
        start = perf_counter()
        with TestClient(server.app) as http:
            startup_seconds = perf_counter() - start
            assert http.get("/health").json() == {"model": "fake-llm"}
            for case in (cases[i] for i in warmup_indices):
                http.post(review(http, case["question"]) + "/reply").raise_for_status()
            rows = measure(http, cases, runs, active)
    raw = asyncio.run(observe_candidates(open_policy_index(runtime), cases))
    for row in rows:
        row.update(raw[row["id"]])
        row["answer_spans_found"] = (
            bool(row.get("spans")) and bool(row["reply"])
            and all(answer_contains_span(span, row["reply"]) for span in row["spans"])
        )

    grouped = {
        "fixed": [row for row in rows if not row["id"].startswith("S")],
        "new_and_mixed": [row for row in rows if row["group"] in ("new", "mixed")],
        "new_only": [row for row in rows if row["group"] == "new"],
        "mixed": [row for row in rows if row["group"] == "mixed"],
        "new_controls": [row for row in rows if row["group"] == "control"],
    }
    if diagnostic_file:
        grouped = {"diagnostic": rows}
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index": {key: value for key, value in manifest.items() if key != "directory"},
        "index_bytes": index_bytes,
        "build_seconds": round(build_seconds, 3),
        "startup_seconds": round(startup_seconds, 3),
        "environment": {pkg: version(pkg) for pkg in ("fastapi", "chromadb", "langgraph", "fastembed")},
        "run": {"repetitions": runs, "warmups": 3, "transport": "FastAPI TestClient", "fake_delay_seconds": 0,
                "openai_requests": 0, "diagnostics_only": diagnostic_file is not None,
                "new_queries_sha256": digest(diagnostic_file or HERE / "questions.json"),
                "course_queries_sha256": digest(Path("data/golden_set.json")),
                "fixed_additions_sha256": digest(HERE.parent / "policy_cases.json"),
                "lock_sha256": digest(Path("uv.lock")), "runner_sha256": digest(Path(__file__))},
        "summary": {name: summarize(subset) for name, subset in grouped.items()},
        "cases": rows,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default="c24a37b")
    parser.add_argument("--expanded-only", action="store_true", help="155조 확장판만 측정합니다.")
    parser.add_argument("--diagnostics-only", action="store_true", help="155조의 추가 진단 문항만 측정합니다.")
    parser.add_argument("--diagnostic-file", type=Path, help="별도 문항 JSON으로 155조를 측정합니다.")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs는 1 이상이어야 합니다.")
    output = args.output.resolve()
    diagnostic_file = args.diagnostic_file
    if args.diagnostics_only:
        diagnostic_file = diagnostic_file or HERE / "diagnostics.json"
    if diagnostic_file:
        diagnostic_file = diagnostic_file.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(output, args.runs, diagnostic_file)
        return
    if args.markdown is None or not args.markdown.read_bytes().startswith(BASE.read_bytes()):
        parser.error("--markdown은 기존 규정 35조를 그대로 포함한 확장 문서를 지정하세요.")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.source_commit}^{{commit}}"], cwd=REPO, text=True,
    ).strip()
    with TemporaryDirectory(prefix="ohgi-scale-") as directory:
        temp = Path(directory)
        archive = temp / "source.zip"
        subprocess.run(["git", "archive", "--format=zip", f"--output={archive}", source_commit], cwd=REPO, check=True)
        results = {}
        corpora = [("expanded", args.markdown.resolve())]
        if not (args.expanded_only or diagnostic_file):
            corpora.insert(0, ("original", None))
        for label, policy in corpora:
            print(f"{label} · {source_commit[:7]}", flush=True)
            results[label] = measure_version(
                source_commit, archive, temp, label, policy, temp / f"{label}.json", args.runs, diagnostic_file,
            )
            print(json.dumps(results[label]["summary"], ensure_ascii=False), flush=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
