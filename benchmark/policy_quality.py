"""Compare the same retrieval, FastAPI and validation cases across server versions."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import median, quantiles
from time import perf_counter
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(Path.cwd()))

from fastapi.testclient import TestClient

from app import main as server
from app.clients.llm_client import FakeLlmClient
from app.core.config import Settings
from app.schemas.reply_draft import GroundedReplyDraft, ReplyVerdict
from app.services.policy_search import PolicySearch


HERE = Path(__file__).resolve().parent
DATA = Path.cwd() / "data"
INCIDENT = "주문한 메뉴와 다른 음식이 왔고 배달 용기가 깨져 국물이 쏟아졌어요."
SWAPPED = "오배송 사진은 수령 1시간 안에, 용기 파손 사진은 수령 24시간 안에 알려 주세요."
INCIDENT_PHOTO = r"(?:오배송|다른 메뉴|이물질)[^.?!]{0,35}24\s*시간[^.?!]{0,25}사진"
SPILL_PHOTO = r"(?:용기 파손|용기가 파손|내용물 유출)[^.?!]{0,35}1\s*시간[^.?!]{0,25}사진"


def load_cases():
    gold = json.loads((DATA / "golden_set.json").read_text(encoding="utf-8"))
    supplied = [
        {"id": f"G{i:02}", "group": "gold", "question": row["question"], "required": [row["source"]]}
        for i, row in enumerate(gold, 1)
    ]
    additions = json.loads((HERE / "policy_cases.json").read_text(encoding="utf-8"))
    return supplied + additions


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def review(http, question):
    created = http.post("/api/reviews", json={"customerName": "실습 손님", "rating": 4, "content": question})
    created.raise_for_status()
    return f"/api/reviews/{created.json()['reviewId']}"


def measure(http, cases, runs, active):
    rows = []
    for case in cases:
        observed = []
        for _ in range(runs):
            path = review(http, case["question"])
            active.clear()
            start = perf_counter()
            response = http.post(path + "/reply")
            elapsed = round((perf_counter() - start) * 1000, 3)
            response.raise_for_status()
            assert len(active) == 1, "답글 API가 정확히 한 번 규정 검색 서비스에 진입해야 합니다."
            result = response.json()
            stored = http.get(path).json()["reply"]
            assert stored == result["reply"], (case["id"], result)
            observed.append({
                **active[0], "api_ms": elapsed, "status": result["status"],
                "answer_articles": result["articleIds"], "attempts": result["attempts"],
                "saved": stored is not None, "reply": stored,
            })
        stable_keys = ("articles", "hold", "reason", "retried_questions", "status", "answer_articles", "attempts")
        assert all(
            all(run[key] == observed[0][key] for key in stable_keys) for run in observed
        ), (case["id"], observed)
        rows.append({**case, **observed[0],
                     "api_samples_ms": [run["api_ms"] for run in observed],
                     "search_samples_ms": [run["ms"] for run in observed]})
    return rows


def capture_search():
    active = []
    original = PolicySearch.retrieve

    async def capture(search, *args, **kwargs):
        start = perf_counter()
        evidence = await original(search, *args, **kwargs)
        active.append({
            "articles": [article.id for article in evidence.articles],
            "hold": evidence.needs_manual_review,
            "reason": getattr(evidence, "reason", ""),
            "retried_questions": list(getattr(evidence, "retried_questions", ())),
            "ms": round((perf_counter() - start) * 1000, 3),
        })
        return evidence

    return active, patch.object(PolicySearch, "retrieve", capture)


def check_gate(http):
    path = review(http, INCIDENT)
    good = http.post(path + "/reply")
    good.raise_for_status()
    reply = good.json()["reply"]
    assert reply and re.search(INCIDENT_PHOTO, reply) and re.search(SPILL_PHOTO, reply), good.json()
    judge_calls = 0

    async def swapped(_self, _state):
        return GroundedReplyDraft(
            reply=SWAPPED, tone="사과", needs_followup=True, article_ids=["제12조", "제17조"],
        )

    async def approves(_self, _state):
        nonlocal judge_calls
        judge_calls += 1
        return ReplyVerdict(passed=True, reason="fixture")

    with patch.object(FakeLlmClient, "draft", swapped), patch.object(FakeLlmClient, "review", approves):
        response = http.post(path + "/reply")
    response.raise_for_status()
    result = response.json()
    return {
        "swapped_status": result["status"],
        "draft_attempts": result["attempts"],
        "model_review_calls": judge_calls,
        "previous_reply_preserved": http.get(path).json()["reply"] == reply,
    }


def check_routing(http):
    probes = (
        ("personal_card_refund", "제 카드 결제 건 환불 들어왔나요?"),
        ("personal_refund_with_damage", "포장 용기가 깨졌는데 제 환불금은 계좌에 들어왔나요?"),
        ("damage_with_no_wrong_delivery", "오배송은 없었고 배달 용기만 깨졌어요."),
        ("card_refund_policy", "카드 결제를 취소하면 환불은 언제 들어오나요?"),
        ("group_menu", "이미 예약한 단체는 메뉴를 미리 골라야 하나요?"),
    )
    outcomes = {}
    for name, question in probes:
        path = review(http, question)
        response = http.post(path + "/reply")
        response.raise_for_status()
        result = response.json()
        assert http.get(path).json()["reply"] == result["reply"]
        outcomes[name] = {
            "question": question, "status": result["status"], "article_ids": result["articleIds"],
            "attempts": result["attempts"], "saved": result["reply"] is not None,
        }
    return outcomes


def summarize(rows):
    cases = [row for row in rows if row["required"]]
    counts = Counter()
    groups = {}
    for row in rows:
        needed = set(row["required"])
        if not needed:
            expected = "ready" if row["id"] == "N01" else "needs_review"
            counts["no_policy_correct"] += row["status"] == expected and not row["answer_articles"]
            counts["unsupported_saved"] += row["id"] != "N01" and row["saved"]
            continue
        group = groups.setdefault(row["group"], {"cases": 0, "search_coverage": 0, "reply_coverage": 0})
        group["cases"] += 1
        group["search_coverage"] += needed.issubset(row["articles"])
        group["reply_coverage"] += row["saved"] and needed.issubset(row["answer_articles"])
        counts["ready_without_all_citations"] += row["saved"] and not needed.issubset(row["answer_articles"])
        counts["policy_hold"] += row["hold"]
        counts["policy_without_articles"] += not row["articles"] and not row["hold"]
    for name in ("search", "api"):
        samples = [sample for row in rows for sample in row[f"{name}_samples_ms"]]
        counts[f"{name}_median_ms"] = round(median(samples), 3)
        counts[f"{name}_p95_ms"] = round(quantiles(samples, n=20, method="inclusive")[18], 3)
    return {
        "cases": len(rows),
        "policy_cases": len(cases),
        "search_coverage": sum(group["search_coverage"] for group in groups.values()),
        "reply_coverage": sum(group["reply_coverage"] for group in groups.values()),
        "groups": groups,
        "retried_requests": sum(bool(row["retried_questions"]) for row in rows),
        **dict(counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--runtime-dir", type=Path, default=Path(".local"))
    parser.add_argument("--source-commit", help="git archive에서 가져온 소스의 40자리 커밋 해시")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs는 1 이상이어야 합니다.")
    if args.source_commit and not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit에는 전체 커밋 해시를 입력하세요.")
    runtime = args.runtime_dir.resolve()
    manifest_path = runtime / "policy-index.json"
    if not manifest_path.is_file():
        parser.error("먼저 python -m app.services.policy_index로 인덱스를 만드세요.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == sha256(DATA / "ohgi_regulations.md")
    cases = load_cases()
    with tempfile.TemporaryDirectory(prefix="ohgi-policy-benchmark-") as temp:
        data = Path(temp)
        settings = Settings(_env_file=None, openai_api_key=None, fake_llm=True, fake_llm_delay_seconds=0,
                            runtime_dir=data, thread_pool_size=None)
        real_index = server.open_policy_index
        override = {"get_settings": lambda: settings, "open_policy_index": lambda _: real_index(runtime)}
        active, observe_search = capture_search()
        with patch.multiple(server, **override), observe_search, TestClient(server.app) as http:
            assert http.get("/health").json()["model"] == "fake-llm"
            for case in (cases[0], cases[13], cases[21]):
                http.post(review(http, case["question"]) + "/reply").raise_for_status()
            rows = measure(http, cases, args.runs, active)
            gate = check_gate(http)
            probes = check_routing(http)
    inputs = {
        "regulations": sha256(DATA / "ohgi_regulations.md"),
        "course_questions": sha256(DATA / "golden_set.json"),
        "extra_questions": sha256(HERE / "policy_cases.json"),
        "runner": sha256(Path(__file__)),
        "lockfile": sha256(Path("uv.lock")),
        "embedding_model": manifest["embedding_model"],
        "indexed_articles": manifest["article_count"],
    }
    result = {
        "git_commit": args.source_commit or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        **{name: version(name) for name in ("fastapi", "chromadb", "langgraph", "langchain-openai")}},
        "run": {"repetitions": args.runs, "openai_generation_requests": 0, "transport": "FastAPI TestClient",
                "fake_model_delay_seconds": 0, "warmups": 3},
        "summary": summarize(rows),
        "validation": gate,
        "routing_probes": probes,
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "validation": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
