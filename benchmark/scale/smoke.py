"""Smoke-test the extended policy index through a local Uvicorn connection."""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import httpx

from run import REPO, digest, link_model_file


def wait_for_server(client, process):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and process.poll() is None:
        try:
            result = client.get("/health")
        except httpx.ConnectError:
            time.sleep(0.15)
            continue
        result.raise_for_status()
        assert result.json() == {"model": "fake-llm"}
        return
    raise RuntimeError("로컬 Uvicorn이 시작되지 않았습니다.")


def review_path(http, question):
    created = http.post("/api/reviews", json={"customerName": "실습 손님", "rating": 4, "content": question})
    created.raise_for_status()
    return f"/api/reviews/{created.json()['reviewId']}"


def stop_server(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.terminate()
    try:
        process.wait(8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def stream_reply(http, path):
    with http.stream("POST", path + "/reply/stream") as response:
        response.raise_for_status()
        assert response.headers["content-type"].startswith("text/event-stream")
        event = ""
        frames = []
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            if line.startswith("data: "):
                frames.append((event, json.loads(line.removeprefix("data: "))))
    return frames


def measure_http(http, expect_new_topics):
    beer = review_path(http, "맥주도 배달돼?")
    frames = stream_reply(http, beer)
    assert frames[-1][0] == "final" and frames[-1][1]["articleIds"] == ["제18조"], frames[-1]
    assert any(event == "draft_delta" for event, _ in frames)
    assert frames[-1][1]["reply"] == http.get(beer).json()["reply"]

    chair = review_path(http, "어린 아기랑 가는데 안전띠 달린 아기 의자가 몇 개 있어요?")
    child_chair = http.post(chair + "/reply").json()
    if expect_new_topics:
        assert child_chair["status"] == "ready" and child_chair["articleIds"] == ["제37조"]
        assert "유아 의자 네 개" in child_chair["reply"]
    else:
        assert child_chair["status"] == "needs_review" and child_chair["attempts"] == 0
    assert http.get(chair).json()["reply"] == child_chair["reply"]
    result = {
        "model": http.get("/health").json()["model"],
        "beer_events": [{"event": name, **data} for name, data in frames if name != "draft_delta"],
        "beer_draft_deltas": sum(name == "draft_delta" for name, _ in frames),
        "child_chair": child_chair,
    }
    if expect_new_topics:
        result.update(measure_new_topics(http))
    return result


def measure_new_topics(http):
    combined = review_path(http, "휠체어로 지하 주차장에서 올라갈 때 어느 승강기를 타요? 유아 의자는 몇 개예요?")
    frames = stream_reply(http, combined)
    final_event, final = frames[-1]
    assert final_event == "final" and final["status"] == "ready", frames[-1]
    assert set(final["articleIds"]) == {"제36조", "제37조"}, final
    assert "A구역 승강기" in final["reply"] and "유아 의자 네 개" in final["reply"]
    assert http.get(combined).json()["reply"] == final["reply"]
    revision = http.post(combined + "/reply/revise", json={"instruction": "한 문장으로 줄여 주세요."})
    revision.raise_for_status()
    revised = revision.json()
    assert revised["status"] == "ready" and set(revised["articleIds"]) == {"제36조", "제37조"}, revised
    assert "A구역 승강기" in revised["reply"] and "유아 의자 네 개" in revised["reply"]
    assert http.get(combined).json()["reply"] == revised["reply"]

    private = review_path(http, "주차 지원이 지금 제 차 번호에 등록됐는지 확인해 주세요.")
    held = http.post(private + "/reply").json()
    assert held["status"] == "needs_review" and held["attempts"] == 0
    assert http.get(private).json()["reply"] is None
    return {"combined_final": final, "combined_revision": revised, "private_lookup": held}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", default="c24a37b")
    parser.add_argument("--expect-new-topics", action="store_true")
    args = parser.parse_args()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.source_commit}^{{commit}}"], cwd=REPO, text=True,
    ).strip()
    with TemporaryDirectory(prefix="ohgi-scale-http-") as directory:
        root = Path(directory)
        archive = root / "app.zip"
        subprocess.run(["git", "archive", "--format=zip", f"--output={archive}", source_commit], cwd=REPO, check=True)
        snapshot = root / "app"
        with ZipFile(archive) as files:
            files.extractall(snapshot)
        shutil.copyfile(args.markdown, snapshot / "data" / "ohgi_regulations.md")
        runtime = snapshot / ".local"
        shutil.copytree(REPO / ".local" / "embedding-models", runtime / "embedding-models", copy_function=link_model_file)
        env = {**os.environ, "PYTHONUTF8": "1", "RUNTIME_DIR": str(runtime), "FAKE_LLM": "true",
               "FAKE_LLM_DELAY_SECONDS": "0.2", "LANGSMITH_TRACING": "false", "HF_HUB_OFFLINE": "1"}
        env.pop("OPENAI_API_KEY", None)
        subprocess.run(
            [sys.executable, "-m", "app.services.policy_index"], cwd=snapshot, env=env, check=True,
            stdout=subprocess.DEVNULL,
        )
        index = json.loads((runtime / "policy-index.json").read_text(encoding="utf-8"))
        assert index["article_count"] == 155 and index["source_sha256"] == digest(args.markdown)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (root / "uvicorn.log").open("wb") as log:
            with subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            ) as process:
                try:
                    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=8, trust_env=False) as http:
                        wait_for_server(http, process)
                        result = measure_http(http, args.expect_new_topics)
                finally:
                    stop_server(process)
        result.update(source_commit=source_commit, articles=index["article_count"], source_sha256=index["source_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
