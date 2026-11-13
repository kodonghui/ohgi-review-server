"""실제 FastAPI 진입점에서 스트리밍, 메모리와 공급자 응답을 검사한다."""

import json
import re
from collections.abc import Iterable
from threading import Event, Thread

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from app import main as app_main
from app.clients.llm_client import FakeLlmClient
from app.core.exceptions import AiRequestError
from app.main import app
from app.schemas.reply_draft import GroundedReplyDraft


def events(lines: Iterable[str]):
    name, data = None, []
    for line in lines:
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").lstrip(" "))
        elif not line and name:
            yield name, json.loads("\n".join(data))
            name, data = None, []


def provider_response(text: str, status: str = "completed") -> dict:
    return {
        "id": "resp_fixture",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-luna",
        "status": status,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "output": [{
            "type": "message",
            "id": "msg_fixture",
            "status": status,
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "none",
    }


def provider_stream(text: str, text_config: dict, *, complete: bool = True) -> bytes:
    frames = [{"type": "response.created", "response": provider_response("", "in_progress")}]
    # 첫 조각은 JSON의 다른 필드가 아닌 답글 문자열 중간에서 자른다.
    midpoint = len('{"reply":"') + 12
    for delta in (text[:midpoint], text[midpoint:]):
        frames.append({
            "type": "response.output_text.delta", "item_id": "msg_fixture", "content_index": 0,
            "output_index": 0, "delta": delta, "logprobs": [],
        })
    if complete:
        frames.append({"type": "response.completed", "response": {**provider_response(text), "text": text_config}})
    return "".join(
        f"event: {frame['type']}\ndata: {json.dumps({**frame, 'sequence_number': i}, ensure_ascii=False)}\n\n"
        for i, frame in enumerate(frames)
    ).encode()


@pytest.fixture
def transport_settings(local_settings, monkeypatch):
    """ChatOpenAI를 실제로 실행하면서 HTTP 전송만 로컬에서 바꿔 끼운다."""
    request_bodies = []
    draft_text = GroundedReplyDraft(
        reply="방문해 주셔서 감사합니다. 소중한 후기를 직원들에게 전하겠습니다.",
        tone="감사", needs_followup=False, article_ids=[],
    ).model_dump_json()

    def enable(handler):
        def transport(request):
            assert request.url.path == "/v1/responses"
            body = json.loads(request.content)
            request_bodies.append(body)
            return handler(body)

        mock = httpx.MockTransport(transport)
        http_clients = (
            httpx.Client(transport=mock),
            httpx.AsyncClient(transport=mock),
        )
        monkeypatch.setattr(app_main, "open_model_http_clients", lambda timeout: http_clients)
        monkeypatch.setattr(app_main, "get_settings", lambda: local_settings.model_copy(update={
            "openai_api_key": SecretStr("local-test-placeholder"),
            "fake_llm": False,
        }))
        return http_clients

    return enable, request_bodies, draft_text


def test_chatopenai_structured_reply_and_token_stream(transport_settings):
    enable, requests, draft_text = transport_settings

    def model(body):
        name = body["text"]["format"]["name"]
        text = draft_text if name == "GroundedReplyDraft" else '{"passed":true,"reason":"검토용 내부 문자열"}'
        if body.get("stream"):
            return httpx.Response(
                200, content=provider_stream(text, body["text"]), headers={"Content-Type": "text/event-stream"},
            )
        return httpx.Response(200, json=provider_response(text))

    http_clients = enable(model)
    with TestClient(app) as client:
        response = client.post("/api/reviews/1/reply")
        assert response.status_code == 200
        assert response.json()["tone"] == "감사"
        sse = client.post("/api/reviews/1/reply/stream")
        assert sse.status_code == 200
        frames = list(events(sse.iter_lines()))
        deltas = [data["text"] for kind, data in frames if kind == "draft_delta"]
        assert len(deltas) >= 2
        assert "".join(deltas) == response.json()["reply"]
        assert frames[-1] == ("final", response.json())
        assert "검토용 내부 문자열" not in sse.text
        assert client.get("/api/reviews/1").json()["reply"] == response.json()["reply"]
    assert all(client.is_closed for client in http_clients)
    assert all(body["store"] is False for body in requests)
    assert any(body.get("stream") for body in requests)
    assert all(not body.get("stream") for body in requests if body["text"]["format"]["name"] == "ReplyVerdict")


def test_stream_eof_before_completed_leaves_existing_reply(transport_settings):
    enable, requests, draft_text = transport_settings

    def model(body):
        if len(requests) == 1:
            return httpx.Response(200, json=provider_response("기존 답글입니다."))
        assert body.get("stream") is True
        return httpx.Response(
            200, content=provider_stream(draft_text, body["text"], complete=False),
            headers={"Content-Type": "text/event-stream"},
        )

    enable(model)
    with TestClient(app) as client:
        assert client.post("/api/reviews/1/reply/sync").status_code == 200
        sse = client.post("/api/reviews/1/reply/stream")
        assert sse.status_code == 200
        frames = list(events(sse.iter_lines()))
        assert any(kind == "draft_delta" for kind, _ in frames)
        assert frames[-1][0] == "error"
        assert frames[-1][1]["code"] == "ERROR_CODE_00011"
        assert client.get("/api/reviews/1").json()["reply"] == "기존 답글입니다."


@pytest.mark.parametrize("status", ["incomplete", "bad_json"])
def test_model_failure_leaves_existing_reply(transport_settings, status):
    enable, requests, draft_text = transport_settings

    def model(body):
        if len(requests) == 1:
            return httpx.Response(200, json=provider_response("기존 답글입니다."))
        output = provider_response(draft_text, "incomplete") if status == "incomplete" else provider_response("{bad")
        return httpx.Response(200, json=output)

    enable(model)
    with TestClient(app) as client:
        assert client.post("/api/reviews/1/reply/sync").status_code == 200
        result = client.post("/api/reviews/1/reply")
        assert result.status_code == 502
        assert result.json()["code"] == "ERROR_CODE_00011"
        assert client.get("/api/reviews/1").json()["reply"] == "기존 답글입니다."


def test_mock_provider_401_keeps_api_error_contract(transport_settings):
    enable, requests, _ = transport_settings
    enable(lambda body: httpx.Response(
        401, json={"error": {"message": "fixture internal marker", "type": "invalid_request_error"}},
    ))
    with TestClient(app) as client:
        response = client.post("/api/reviews/1/reply")
        assert response.status_code == 502
        assert response.json()["code"] == "ERROR_CODE_00011"
        assert "fixture internal marker" not in response.text
        assert client.get("/api/reviews/1").json()["reply"] is None
    assert len(requests) == 1


def test_sse_maps_failure_after_headers_to_error_and_releases_review(client, monkeypatch):
    async def fail(self, state):
        raise AiRequestError("모의 공급자 장애")

    assert client.post("/api/reviews/1/reply").status_code == 200
    previous = client.get("/api/reviews/1").json()["reply"]
    with monkeypatch.context() as patch:
        patch.setattr(FakeLlmClient, "draft", fail)
        response = client.post("/api/reviews/1/reply/stream")
    frames = list(events(response.iter_lines()))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert frames[-1] == ("error", {"code": "ERROR_CODE_00011", "detail": "모의 공급자 장애"})
    assert all(kind != "final" for kind, _ in frames)
    assert client.get("/api/reviews/1").json()["reply"] == previous
    assert client.post("/api/reviews/1/reply").status_code == 200
    assert client.post("/api/reviews/99/reply/stream").status_code == 404


def test_revision_reads_only_the_same_review_and_store_survives_restart(local_settings, monkeypatch):
    seen = []
    original = FakeLlmClient.draft

    async def observe(self, state):
        seen.append((state["review_text"], state.get("previous_reply"), state["preferences"].style))
        return await original(self, state)

    monkeypatch.setattr(FakeLlmClient, "draft", observe)
    revision = {"instruction": "한 문장으로 줄여 주세요."}
    with TestClient(app) as client:
        preferences = {"style": "따뜻", "maxSentences": 3}
        assert client.put("/api/reply-preferences", json=preferences).json() == preferences
        first = client.post("/api/reviews/1/reply").json()
        second = client.post("/api/reviews/2/reply").json()
        assert client.post("/api/reviews/3/reply/revise", json=revision).status_code == 409
        revised = client.post("/api/reviews/1/reply/revise", json=revision).json()
        assert seen[-1][1] == first["reply"] != second["reply"]
        assert seen[-1][2] == "따뜻"
        assert len(re.split(r"(?<=[.!?])\s+", revised["reply"])) == 1
        assert client.get("/api/reviews/1").json()["reply"] == revised["reply"]
        assert client.post("/api/reviews/1/reply/revise", json={"instruction": "  "}).status_code == 422

    with TestClient(app) as restarted:
        assert restarted.get("/api/reply-preferences").json() == preferences
        assert restarted.get("/api/reviews/1").json()["reply"] is None
        assert restarted.post("/api/reviews/1/reply/revise", json=revision).status_code == 409


def test_concurrent_request_to_same_review_gets_409(client, monkeypatch):
    import asyncio

    entered = Event()
    release = Event()
    original = FakeLlmClient.draft

    async def hold(self, state):
        if state["customer_name"] == "김하늘":
            entered.set()
            await asyncio.to_thread(release.wait, 3)
        return await original(self, state)

    monkeypatch.setattr(FakeLlmClient, "draft", hold)
    responses = []
    worker = Thread(target=lambda: responses.append(client.post("/api/reviews/1/reply")))
    worker.start()
    try:
        assert entered.wait(3)
        assert client.post("/api/reviews/1/reply/stream").status_code == 409
        assert client.post("/api/reviews/1/reply/sync").status_code == 409
        assert client.post("/api/reviews/2/reply").status_code == 200
    finally:
        release.set()
        worker.join(timeout=3)
    assert responses[0].status_code == 200
