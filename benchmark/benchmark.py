"""
답글 API 두 버전의 처리량 비교 (심화 : 동기 버전과 비동기 버전)

Postman 탭 두 개로 본 차이를 동시 요청 100개로 잰다.
- 1) /api/reviews/1/reply/sync : def 핸들러 + 동기 메서드. 스레드 풀(기본 40개)의 스레드가 요청을 하나씩 맡아 기다린다.
- 2) /api/reviews/1/reply      : async def 핸들러 + await. 이벤트 루프 하나가 요청 100개를 함께 기다린다.

실제 OpenAI로 200번 요청하면 그만큼 비용이 들기 때문에, 서버를 FAKE_LLM=true로 실행했을 때만 잰다.
가짜 LLM은 응답 시간(1초)이 늘 같아서 서버가 기다리는 방법의 차이만 볼 수 있다.

[지표]
- p50 : 응답 시간을 줄 세웠을 때 가운데 값. 보통의 사용자가 기다린 시간
- p95 : 95%의 요청이 이 시간 안에 끝났다는 뜻. 느린 쪽 사용자가 기다린 시간
- RPS : 1초에 끝낸 요청 수 (처리량)

[실행 방법] 터미널 두 개를 연다. (Windows PowerShell)
  터미널 1 : $env:FAKE_LLM="true"; $env:FAKE_LLM_DELAY_SECONDS="1"; uv run fastapi dev
  터미널 2 : uv run python benchmark/benchmark.py
  (macOS는 터미널 1에서 FAKE_LLM=true FAKE_LLM_DELAY_SECONDS=1 uv run fastapi dev)
  다 쟀으면 터미널 1을 Ctrl+C로 끈다. 그 터미널에는 환경 변수가 남아 있으니, 평소처럼 실행할 때는 새 터미널을 연다.
"""

import asyncio
import statistics
import time

import httpx

# Windows에서 localhost는 IPv6(::1)를 먼저 시도하느라 연결마다 지연이 생길 수 있어서 127.0.0.1을 쓴다.
BASE_URL = "http://127.0.0.1:8000"

# (이름, 주소, 동시에 보낼 요청 수). 같은 리뷰(1번)에 답글을 계속 덮어쓴다.
TARGETS = [
    ("1) reply/sync", "/api/reviews/1/reply/sync", 100),
    ("2) reply", "/api/reviews/1/reply", 100),
]
# 요청을 나눠 보낼 클라이언트 수. 클라이언트 하나로 100개를 보내면 측정 도구 쪽이 먼저 느려진다(chap03).
CLIENT_COUNT = 5


async def is_fake_server(client: httpx.AsyncClient) -> bool:
    """
    가짜 LLM으로 실행한 서버인지 요청 한 번으로 먼저 확인한다.
    실제 OpenAI로 실행한 서버에 요청 200개를 보내는 사고를 막는 안전장치다. (그 서버라면 이 확인 요청 한 번만 OpenAI로 간다)
    """

    response = await client.post("/api/reviews/1/reply")
    return response.status_code == 200 and response.json().get("model") == "fake-llm"


async def send_one(client: httpx.AsyncClient, path: str) -> tuple[float, bool]:
    """요청 하나를 보내고 (걸린 시간(초), 성공 여부)를 반환한다."""

    start = time.perf_counter()
    try:
        response = await client.post(path)
        ok = response.status_code == 200
    except httpx.HTTPError:
        ok = False
    return time.perf_counter() - start, ok


async def measure(clients: list[httpx.AsyncClient], name: str, path: str, count: int) -> None:
    """count개의 요청을 한꺼번에 보내고 지표를 출력한다."""

    start = time.perf_counter()

    # asyncio.gather : 요청 count개를 동시에 보내고, 모두 끝날 때까지 함께 기다린다.
    results = await asyncio.gather(*(send_one(clients[i % CLIENT_COUNT], path) for i in range(count)))

    total = time.perf_counter() - start
    latencies = sorted(elapsed for elapsed, _ in results)
    failures = sum(1 for _, ok in results if not ok)

    p50 = statistics.median(latencies)
    # quantiles(n=20)는 5% 간격의 경계값 19개를 반환한다. 19번째(인덱스 18)가 p95다.
    p95 = statistics.quantiles(latencies, n=20, method="inclusive")[18]

    print(f"{name:14s} 요청 {count:3d}개 | 전체 {total:6.2f}초 | p50 {p50:5.2f}초 | p95 {p95:5.2f}초 | "
          f"최대 {latencies[-1]:5.2f}초 | RPS {count / total:6.1f} | 실패 {failures}")


async def main() -> None:
    clients = [httpx.AsyncClient(base_url=BASE_URL, timeout=120) for _ in range(CLIENT_COUNT)]
    try:
        if not await is_fake_server(clients[0]):
            print("FAKE_LLM=true로 실행한 서버가 아니라서 측정을 중단한다. 맨 위의 [실행 방법]을 확인한다.")
            return

        for name, path, count in TARGETS:
            await measure(clients, name, path, count)
    finally:
        for client in clients:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
