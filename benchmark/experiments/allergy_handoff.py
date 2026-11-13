"""Prototype staff-only evidence attachment on the original 35-article corpus."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from policy_quality import load_cases, measure, server, sha256, summarize

from app.core.config import Settings
from app.services.policy_search import PolicySearch, PolicySearchResult, SEARCH_ERRORS, valid_article
from app.services.policy_topics import PolicyEvidence
from fastapi.testclient import TestClient


ALLERGY = re.compile(r"알레르기|알러지|교차\s*(?:오염|접촉)")
ARTICLE_ID = "제20조"
SOURCE_PATTERN = re.compile(r"같은\s*주방[^.]{0,45}미량이\s*섞일\s*가능성[^.]{0,20}없앨\s*수는\s*없다\.")
BASE_COMMIT = "b3c7dba"

HOLDOUT = (
    ("H01", "우유 알러지가 있는데 이 메뉴 먹어도 돼요?", [ARTICLE_ID]),
    ("H02", "같은 조리도구를 쓰면 교차 접촉 위험은 없나요?", [ARTICLE_ID]),
    ("H03", "땅콩 알레르기가 있어서 견과 토핑을 빼 달라고 했는데 제 주문에 반영됐나요?", [ARTICLE_ID]),
    ("H04", "알레르기가 있어요. 어떤 메뉴가 저한테 안전한지 추천해 주세요.", [ARTICLE_ID]),
    ("H05", "배달 용기가 깨졌어요. 알레르기도 있는데 이걸 먹어도 돼요?", ["제17조", ARTICLE_ID]),
    ("H06", "견과류에 심한 반응이 있는데 이 소스를 먹어도 되나요?", [ARTICLE_ID]),
    ("H07", "제 카드 결제 건 환불 들어왔나요?", []),
    ("H08", "식사 중에 안내해 주셔서 감사해요.", []),
)


def handoff_with_evidence(original):
    async def retrieve(search, text):
        if not ALLERGY.search(text):
            return await original(search, text)

        held = PolicySearchResult((ARTICLE_ID,), (), True, reason="staff_allergy_evidence_missing")
        if search._index is None:
            return held
        try:
            articles = await search._index.aget_by_ids([ARTICLE_ID])
        except SEARCH_ERRORS:
            return held
        for article in articles:
            if article.id != ARTICLE_ID or not valid_article(article):
                continue
            if match := SOURCE_PATTERN.search(article.page_content):
                evidence = PolicyEvidence("allergy_handoff", ARTICLE_ID, (match.group(0),), None, True)
                return PolicySearchResult(
                    (ARTICLE_ID,), (article,), True, evidence=(evidence,), reason="staff_allergy_with_evidence",
                )
        return held

    return retrieve


def run_case_set(cases, index, variant):
    active = []
    original = PolicySearch.retrieve
    candidate = handoff_with_evidence(original) if variant else original

    async def trace(search, text):
        start = perf_counter()
        evidence = await candidate(search, text)
        active.append({
            "articles": [article.id for article in evidence.articles],
            "hold": evidence.needs_manual_review,
            "reason": evidence.reason,
            "retried_questions": list(evidence.retried_questions),
            "handoff_excerpts": [line for item in evidence.evidence if item.topic == "allergy_handoff" for line in item.excerpts],
            "ms": round((perf_counter() - start) * 1000, 3),
        })
        return evidence

    with TemporaryDirectory(prefix="ohgi-allergy-") as directory:
        runtime = Path(directory)
        settings = Settings(
            _env_file=None, openai_api_key=None, fake_llm=True, fake_llm_delay_seconds=0, runtime_dir=runtime,
        )
        with patch.multiple(server, get_settings=lambda: settings, open_policy_index=lambda _: index):
            with patch.object(PolicySearch, "retrieve", trace), TestClient(server.app) as http:
                assert http.get("/health").json() == {"model": "fake-llm"}
                return measure(http, cases, 3, active)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(["git", "rev-parse", BASE_COMMIT], text=True).strip()
    subprocess.run(["git", "diff", "--quiet", source, "--", "app", "data", "uv.lock"], check=True)
    cases = load_cases()
    probes = [
        {"id": name, "group": "held_out", "question": question, "required": required}
        for name, question, required in HOLDOUT
    ]
    index = server.open_policy_index(REPO / ".local")
    assert index is not None
    measurements = {}
    for name, variant in (("current", False), ("prototype", True)):
        rows = run_case_set(cases + probes, index, variant)
        fixed = rows[:len(cases)]
        extra = rows[len(cases):]
        g15 = next(row for row in fixed if row["id"] == "G15")
        measurements[name] = {"fixed": summarize(fixed), "g15": g15, "probes": extra, "cases": fixed}

    current = measurements["current"]
    prototype = measurements["prototype"]
    assert current["g15"]["articles"] == []
    assert prototype["g15"]["articles"] == [ARTICLE_ID]
    assert prototype["fixed"]["search_coverage"] == 44 and prototype["fixed"]["reply_coverage"] == 43
    for row in [prototype["g15"], *prototype["probes"][:-1]]:
        assert row["status"] == "needs_review" and row["attempts"] == 0 and not row["saved"], row
    assert prototype["probes"][-1]["status"] == "ready"
    no_index = run_case_set([cases[14]], None, True)[0]
    assert no_index["reason"] == "staff_allergy_evidence_missing" and not no_index["saved"]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source_commit": source,
        "source_sha256": sha256(REPO / "data" / "ohgi_regulations.md"),
        "runner_sha256": sha256(Path(__file__)),
        "openai_requests": 0,
        "per_case_repetitions": 3,
        "no_index_probe": no_index,
        **measurements,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, measurement in measurements.items():
        print(name, json.dumps(measurement["fixed"], ensure_ascii=False))
        for row in [measurement["g15"], *measurement["probes"]]:
            print(row["id"], row["status"], row["reason"], row["articles"], row["attempts"])


if __name__ == "__main__":
    main()
