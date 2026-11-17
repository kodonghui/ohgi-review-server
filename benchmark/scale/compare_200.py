"""Compare two executions of the same LangSmith benchmark without making model calls."""

import argparse
import json
import re
from collections import Counter
from decimal import Decimal
from urllib.parse import urlencode

from benchmark.langsmith_eval import langsmith_client
from benchmark.scale.eval_200 import EMBED_PRICE, MIL, experiment_name, read_journal, usage_receipt
from benchmark.scale.index_200 import ARTIFACTS, write_json
from benchmark.scale.report_200 import (
    ENGINES, PROVIDERS, REPORT, case_trace, collect, dataset_url, dollar, read, table, write_study_html,
)


TITLE = "LangGraph 재설계 전후 답글 API 벤치마크"
SOURCE_PATHS = ("app/services/reply_workflow.py", "app/services/policy_search.py", "app/services/policy_topics.py",
                "app/services/policy_passages.py", "app/clients/llm_client.py", "benchmark/scale/eval_200.py")


def read_variant(directory, variant, golden, before):
    after = {}
    for (engine, label), baseline in before.items():
        name = experiment_name(golden, engine, label, variant)
        path = directory / engine / f"{name}.json"
        run = read(path)
        ignore = {"experiment", "code_commit"}
        old, new = (item["configuration"] for item in (baseline, run))
        assert {k: v for k, v in old.items() if k not in ignore} == {
            k: v for k, v in new.items() if k not in ignore
        }, f"Configuration mismatch in {name}"
        old_rows, new_rows = (item["rows"] for item in (baseline, run))
        assert Counter(row["case_id"] for row in old_rows) == Counter(row["case_id"] for row in new_rows)
        assert len({row["run_id"] for row in new_rows}) == len(new_rows) == run["summary"]["requests"]
        assert dataset_url(run) == dataset_url(baseline) and run["project_id"] != baseline["project_id"]
        run["cost"] = usage_receipt(path.with_suffix(".usage.jsonl"), path.with_suffix(".embeddings.jsonl"), new["model"])
        after[engine, label] = run
    assert len({run["configuration"]["code_commit"] for run in after.values()}) == 1
    return after


def verify_langsmith(directory, runs):
    client = langsmith_client()
    dataset_id = dataset_url(next(iter(runs.values()))).rsplit("/", 1)[-1]
    examples = {row.id: row.inputs["case_id"] for row in client.list_examples(dataset_id=dataset_id)}
    assert len(examples) == 32
    receipt = []
    for (engine, label), run in runs.items():
        project = client.read_project(project_id=run["project_id"])
        assert str(project.reference_dataset_id) == dataset_id
        roots = list(client.list_runs(project_id=project.id, is_root=True,
                                      select=["id", "end_time", "reference_example_id", "outputs"]))
        expected = {row["run_id"]: row["case_id"] for row in run["rows"]}
        graded = {str(row.id): examples[row.reference_example_id] for row in roots
                  if row.end_time and str(row.id) in expected}
        assert expected == graded, f"LangSmith roots differ in {engine}/{label}"
        ungraded = [str(row.id) for row in roots if str(row.id) not in graded]
        assert not any(root.outputs for root in roots if str(root.id) in ungraded)
        receipt.append({"engine": engine, "label": label, "count": len(graded),
                        "project_id": run["project_id"], "ungraded_roots": ungraded})
        print(engine, label, len(graded), "LangSmith 결과 확인", flush=True)
    write_json(directory / "langsmith-verification.json", {"dataset_id": dataset_id, "experiments": receipt})


def compare_url(before, after):
    params = {"selectedSessions": f"{before['project_id']},{after['project_id']}", "source": before["project_id"]}
    return f"{dataset_url(before)}/compare?{urlencode(params)}"


def change(before, after, key, denominator=None):
    suffix = f"/{denominator}" if denominator else ""
    return f"{before[key]}{suffix} → {after[key]}{suffix}"


def case_passes(run):
    return Counter(row["case_id"] for row in run["rows"] if row["passed"])


def reply_failures(run, expected):
    complete = [row for row in run["rows"] if row.get("http_status") == 200]
    return (
        len(complete),
        sum(not row["articles_match"] for row in complete),
        sum(not row["notices_match"] for row in complete),
        sum(row.get("status") == "needs_review" and expected[row["case_id"]]["status"] == "ready"
            for row in complete),
    )


def failure_sample(runs, case_id, notice):
    for run in runs.values():
        for row in run["rows"]:
            if (row["case_id"] == case_id and row.get("reply")
                    and notice in row.get("missing_notices", [])):
                return run, row
    return None


def render(directory, golden, old_lesson, old_api, new_lesson, new_api, variant):
    dataset = dataset_url(next(iter(old_lesson.values())))
    expected = {case["id"]: case["expected"] for case in golden["cases"]}
    old_passed = sum(r["summary"]["passed"] for r in old_api.values())
    new_passed = sum(r["summary"]["passed"] for r in new_api.values())
    requests = sum(r["summary"]["requests"] for r in new_api.values())
    old_approvals = sum(r["summary"]["incorrect_auto_approvals"] for r in old_api.values())
    new_approvals = sum(r["summary"]["incorrect_auto_approvals"] for r in new_api.values())
    old_errors = sum(r["summary"]["http_errors"] for r in old_api.values())
    new_errors = sum(r["summary"]["http_errors"] for r in new_api.values())
    check_tokens = sum(row["tokens"] for row in read_journal(directory / "development.embeddings.jsonl"))
    check_usd = Decimal(check_tokens) * EMBED_PRICE / MIL
    lesson_usd = sum(Decimal(r["cost"]["measured_usd"]) for r in new_lesson.values())
    api_usd = sum(Decimal(r["cost"]["measured_usd"]) for r in new_api.values())
    old_api_usd = sum(Decimal(r["cost"]["measured_usd"]) for r in old_api.values())
    total = check_usd + lesson_usd + api_usd
    old_commit = next(iter(old_api.values()))["configuration"]["code_commit"]
    commit = next(iter(new_api.values()))["configuration"]["code_commit"]
    lesson_rows, api_rows, failures, cases, cache = [], [], [], [], []
    metrics = {"lesson": [], "api": [], "costs": {"lesson": dollar(lesson_usd), "api": dollar(api_usd),
               "development_embeddings": dollar(check_usd), "old_api": dollar(old_api_usd), "total": dollar(total)},
               "requests": requests, "old_passed": old_passed, "new_passed": new_passed,
               "old_approvals": old_approvals, "new_approvals": new_approvals,
               "old_errors": old_errors, "new_errors": new_errors, "variant": variant, "code_commit": commit}
    for (engine, label), new in new_lesson.items():
        old = old_lesson[engine, label]
        a, b = old["summary"], new["summary"]
        url = compare_url(old, new)
        old_rank = {row["case_id"]: row.get("sources") for row in old["rows"]}
        rank_matches = sum(row.get("sources") == old_rank[row["case_id"]] for row in new["rows"])
        lesson_rows.append([ENGINES[engine], f"[{label.replace('k', 'k=')}]({url})",
                            change(a, b, "retrieval_hit", 24), change(a, b, "correct", 32),
                            f"{rank_matches}/32", dollar(new["cost"]["measured_usd"])])
        metrics["lesson"].append(dict(engine=engine, label=label, old=a, new=b, cost=new["cost"],
                                      compare_url=url, rank_matches=rank_matches, project=new["project_id"]))
    for (engine, label), new in new_api.items():
        old = old_api[engine, label]
        a, b = old["summary"], new["summary"]
        url = compare_url(old, new)
        name = f"{ENGINES[engine]} {PROVIDERS[label]}"
        api_rows.append([ENGINES[engine], f"[{PROVIDERS[label]}]({url})", change(a, b, "passed", 96),
                         change(a, b, "all_repetitions_passed", 32), change(a, b, "incorrect_auto_approvals"),
                         change(a, b, "http_errors"),
                         f"{a['median_reply_ms']/1000:.2f}/{a['p95_reply_ms']/1000:.2f} → "
                         f"{b['median_reply_ms']/1000:.2f}/{b['p95_reply_ms']/1000:.2f}",
                         f"{dollar(old['cost']['measured_usd'])} → {dollar(new['cost']['measured_usd'])}"])
        failures.append([name, *(f"{a} → {b}" for a, b in zip(
            reply_failures(old, expected), reply_failures(new, expected), strict=True,
        ))])
        old_cases, new_cases = case_passes(old), case_passes(new)
        cases.append([name, sum(new_cases[k] > old_cases[k] for k in expected),
                      sum(new_cases[k] < old_cases[k] for k in expected)])
        model_usage = new["cost"]["model_usage"]
        cache.append([name, model_usage.get("calls_with_usage", 0), model_usage.get("cache_read", 0),
                      model_usage.get("cache_write", 0), new["cost"]["query_embedding_usage"]["tokens"]])
        metrics["api"].append(dict(engine=engine, provider=label, old=a, new=b, old_cost=old["cost"],
                                   cost=new["cost"], compare_url=url, project=new["project_id"]))

    texts = [
        "# " + TITLE, "",
        f"① **전체 API.** [실행] 같은 {requests}회에서 통과가 {old_passed}회에서 {new_passed}회로 달라졌습니다. "
        f"직원 확인 문의의 자동 승인은 {old_approvals}→{new_approvals}회, HTTP 오류는 {old_errors}→{new_errors}회입니다.",
        "",
        f"② **비용.** [실행] 같은 문서 벡터를 재사용했습니다. 교재 평가와 전체 API, 개발 질의 임베딩의 합계는 **{dollar(total)} USD**입니다.",
        "",
        f"공통 골든셋 {len(expected)}문항, 605개 조문, 모델·캐시·요청 시간 설정과 코드 채점기를 고정했습니다. "
        f"기준 실행 코드 {old_commit[:7]}와 새 실행 코드 {commit[:7]}를 비교했습니다. "
        f"[공통 LangSmith 데이터셋]({dataset}) · [연구와 설계](REDESIGN_200.md)", "",
        "## 교재의 k=1·3", "",
        "이 단계는 벡터 검색과 GPT 답변·평가를 실행합니다. 검색 적중은 필수 조문을 모두 찾은 수이며, 순서 일치는 같은 저장소와 k의 전후 목록을 뜻합니다.",
        "",
        table(["저장소", "k", "조문 적중", "답변 정확", "전후 조문 순서 일치", "재실험 USD"], lesson_rows), "",
        "## 실제 답글 작성·검토·저장", "",
        table(["저장소", "모델", "통과", "3회 모두 통과", "오승인", "HTTP 오류", "답글 p50/p95 초", "모델·질의 USD"], api_rows), "",
        "모델 ID와 출력 상한, GPT·Claude의 캐시, DeepSeek의 모델 호출당 180초 제한을 유지했습니다. "
        "답글 지연은 FastAPI TestClient의 POST 요청에서 측정했습니다. 문서 검색 수와 응답 길이가 달라지면 비용과 지연도 바뀝니다.",
        "",
        table(["저장소·모델", "HTTP 200", "조문 불일치", "안내 문구 누락", "안내 가능 문의의 보류"], failures), "",
        "이 표는 HTTP 200에 대한 집계입니다. 한 응답이 여러 조건에서 탈락할 수 있습니다.",
        "",
        table(["저장소·모델", "통과 반복이 늘어난 문항", "통과 반복이 줄어든 문항"], cases), "",
        "## 핵심 질문의 전후 트레이스", "",
    ]
    for case in ("M01", "H01", "O05", "M04"):
        rows = []
        for (engine, label), new in new_api.items():
            old = old_api[engine, label]
            outcomes = [row for row in new["rows"] if row["case_id"] == case]
            cited = next((row for row in outcomes if row["passed"]), outcomes[0])
            detail = cited.get("status") or f"HTTP {cited.get('http_status')}"
            if cited.get("article_ids"):
                detail += " · " + ", ".join(cited["article_ids"])
            rows.append([f"{ENGINES[engine]} {PROVIDERS[label]}", f"{case_passes(old)[case]}/3 → {case_passes(new)[case]}/3",
                         f"[{detail}]({case_trace(new, cited)})"])
        texts.extend([f"### {case}", "", table(["저장소·모델", "전후 통과", "재실험 트레이스"], rows), ""])
    case_proof = directory / "verified-case-paths.json"
    if case_proof.exists():
        paths = {row["case_id"]: row["steps"] for row in read(case_proof)}
        assert paths["H01"]["screen_request"]["status"] == "needs_review" and "retrieve" not in paths["H01"]
        assert set(paths["M01"]["retrieve"]["ids"]) == {"제1조", "제6조"}
        assert "article_reference" in paths["O05"]["retrieve"]["retried"]
        texts.extend([
            "Chroma·Claude의 H01 트레이스에서는 직원 확인 노드에서 종료했습니다. M01은 두 질문의 근거를 제1조와 제6조로 나눴습니다. "
            "O05의 규정 검색은 후보에 없던 제69조를 온전히 읽기 위해 제201조의 명시적 참조를 따라갔습니다.", "",
        ])
    evidence_path = directory / "evidence-verification.json"
    if evidence_path.exists():
        evidence = read(evidence_path)
        totals = {
            phase: {
                key: sum(row[phase][key] for row in evidence.values())
                for key in ("staff", "staff_held_in_code", "public_with_expected_ids", "public_supported")
            }
            for phase in ("baseline", "redesign")
        }
        before, after = totals["baseline"], totals["redesign"]
        metrics["evidence_audit"] = totals
        texts.extend([
            f"[실행] GPT·Claude의 두 환경 트레이스 384건에서 작성 전에 필요한 조문을 확보해 진행한 건수는 "
            f"{before['public_supported']}/{before['public_with_expected_ids']} → "
            f"{after['public_supported']}/{after['public_with_expected_ids']}였습니다. "
            f"직원 확인 문항이 모델 호출 전에 보류된 건수는 "
            f"{before['staff_held_in_code']}/{before['staff']} → {after['staff_held_in_code']}/{after['staff']}였습니다.", "",
        ])
    o07 = failure_sample(new_api, "O07", "앱 포인트 내역 확인")
    o05 = failure_sample(new_api, "O05", "가족 대리 수령 가능")
    m01 = failure_sample(new_api, "M01", "전화 주문 불가")
    if o07 and o05:
        texts.extend([
            "## 남은 오류", "",
            "- O07은 올바른 제112조를 선택한 답글 중에도 앱 경로를 빠뜨린 사례가 있습니다. "
            "제112조에서 앱 표시를 설명한 문장과 날짜별 주문번호를 설명한 문장이 나뉩니다. "
            f"[제112조를 인용했지만 앱 경로가 빠진 답글]({case_trace(*o07)})",
            "- O05는 가족에게 주문번호를 전달하면 된다고 안내해도 '가능' 같은 정규식 일치 표현이 없으면 안내 문구 점수를 잃는 사례가 있습니다. "
            f"[대리 수령 안내가 있었던 탈락 답글]({case_trace(*o05)})",
            "",
        ])
    if m01 and "전화가 아닌" in m01[1]["reply"]:
        texts.extend([
            "- M01의 OVH·GPT 답글은 전화 대신 앱이나 홈페이지를 사용하라고 안내했습니다. "
            "채점기의 불가 표현 정규식에 '전화가 아닌'이 없어 1회가 탈락했습니다. 공식 점수는 기존 채점기로 남겼습니다. "
            f"[M01 채점과 실제 답글]({case_trace(*m01)})", "",
        ])
    audit_path = directory / "m06-verdict-audit.json"
    if audit_path.exists():
        audit = read(audit_path)
        assert "제397조" in audit["articles"] and "ChatOpenAI" in audit["review_inputs_containing_article"]
        audit_run = next(run for run in new_api.values() if run["project_id"] == audit["project_id"])
        reference = case_trace(audit_run, audit)
        texts.extend([
            "- M06의 Chroma·GPT 검토 모델은 제397조와 글 리뷰 기준 원문을 입력받고도 제397조가 없다고 판정했습니다. "
            "그 피드백을 받은 재작성은 조문 ID 검사에 걸려 직원 확인으로 끝났습니다. "
            f"[검토 모델 입력과 보류 트레이스]({reference})", "",
        ])
    deepseek_errors = []
    for (engine, label), new in new_api.items():
        if label != "deepseek":
            continue
        errors = Counter(row.get("error_detail") or f"HTTP {row.get('http_status')}"
                         for row in new["rows"] if row.get("http_status") != 200)
        deepseek_errors.extend([ENGINES[engine], reason.replace("|", " "), count] for reason,count in errors.items())
    if deepseek_errors:
        texts.extend(["### DeepSeek 요청 오류", "",
                      table(["저장소", "API 오류 상세", "건수"], deepseek_errors), ""])
    texts.extend([
        "## 사용량", "",
        table(["단계", "기록된 내용", "재실험 USD"], [
            ["공통 문서 임베딩", "보존한 벡터 재사용", "0.00000000"],
            ["교재 k 비교", f"{sum(r['summary']['requests'] for r in new_lesson.values())}회", dollar(lesson_usd)],
            ["전체 API", f"{requests}회", dollar(api_usd)],
            ["개발용 검색", f"{check_tokens} 질의 임베딩 토큰", dollar(check_usd)],
            ["합계", "보고된 사용량", dollar(total)],
        ]), "",
        table(["저장소·모델", "사용량이 있는 모델 응답", "캐시 읽기 토큰", "캐시 쓰기 토큰", "질의 임베딩 토큰"], cache), "",
        "단가는 기준 실험의 표를 사용했습니다. DeepSeek 모델은 릴레이 무료 할당량으로 산정했고, 질의 임베딩을 합산했습니다. "
        "LangSmith·OVH 요금과 응답에 사용량이 없는 호출은 별도입니다. [기준 비용 기록](RESULTS_200.md)", "",
        "## 비교 범위", "",
        "재설계 과정에서 같은 문항과 기준 실패를 검토했으므로 이번 점수는 회귀 평가입니다. "
        "같은 구조 변경에 라우팅, 근거 선택, 검토 프롬프트가 함께 들어갔습니다. 개별 변경 효과와 처음 보는 질문의 성능은 각각 별도 평가로 입증해야 합니다.", "",
    ])
    verification = directory / "langsmith-verification.json"
    if verification.exists():
        proof = read(verification)
        assert proof["dataset_id"] == dataset.rsplit("/", 1)[-1]
        count = sum(r["count"] for r in proof["experiments"])
        texts.extend([f"[실행] {len(proof['experiments'])}개 실험의 완료 트레이스 {count}건을 "
                      "원시 결과의 문항과 실행 ID에 대조했습니다.", ""])
    cleanup = directory / "ovh-cleanup.json"
    if cleanup.exists():
        proof = read(cleanup)
        assert proof["directory_absent"] and proof["postgres_stopped"]
        texts.extend(["[실행] OVH 원본 파일을 회수하고 해시를 검증한 뒤 실험용 PostgreSQL과 임시 디렉터리를 정리했습니다.", ""])
    texts.extend(["## 확인 질문", "",
                  "O07에서 제112조를 이미 찾았는데도 안내가 빠진 까닭을 검색된 조문과 모델에 전달한 원문의 차이로 설명해 보세요.", ""])
    result = "\n".join(texts)
    REPORT.with_name("REDESIGN_RESULTS_200.md").write_text(result, encoding="utf-8")
    study = REPORT.with_name("REDESIGN_200.md").read_text(encoding="utf-8")
    write_study_html(result + "\n---\n\n" + study, commit, title=TITLE, output=directory / "redesign-study-200p.html",
                     paths=SOURCE_PATHS)
    write_json(directory / "summary.json", metrics)
    print(json.dumps({k: v for k, v in metrics.items() if k not in {"lesson", "api"}}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="graph-v2")
    parser.add_argument("--verify-langsmith", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.variant):
        parser.error("Invalid --variant")
    directory = ARTIFACTS / args.variant
    _, golden, old_lesson, old_api = collect()
    new = read_variant(directory, args.variant, golden, old_lesson | old_api)
    if args.verify_langsmith:
        verify_langsmith(directory, new)
    render(directory, golden, old_lesson, old_api, {key: new[key] for key in old_lesson},
           {key: new[key] for key in old_api}, args.variant)


if __name__ == "__main__":
    main()
