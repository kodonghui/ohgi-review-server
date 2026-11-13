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


def measure_http(http):
    beer = review_path(http, "맥주도 배달돼?")
    with http.stream("POST", beer + "/reply/stream") as response:
        response.raise_for_status()
        assert response.headers["content-type"].startswith("text/event-stream")
        event = ""
        frames = []
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            if line.startswith("data: "):
                frames.append((event, json.loads(line.removeprefix("data: "))))
    assert frames[-1][0] == "final" and frames[-1][1]["articleIds"] == ["제18조"], frames[-1]
    assert any(event == "draft_delta" for event, _ in frames)
    assert frames[-1][1]["reply"] == http.get(beer).json()["reply"]

    chair = review_path(http, "어린 아기랑 가는데 안전띠 달린 아기 의자가 몇 개 있어요?")
    held = http.post(chair + "/reply").json()
    assert held["status"] == "needs_review" and held["attempts"] == 0
    assert http.get(chair).json()["reply"] is None
    return {
        "model": http.get("/health").json()["model"],
        "beer_events": [{"event": name, **data} for name, data in frames if name != "draft_delta"],
        "beer_draft_deltas": sum(name == "draft_delta" for name, _ in frames),
        "child_chair": held,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_commit = subprocess.check_output(["git", "rev-parse", "c24a37b"], cwd=REPO, text=True).strip()
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
                        result = measure_http(http)
                finally:
                    stop_server(process)
        result.update(source_commit=source_commit, articles=index["article_count"], source_sha256=index["source_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
