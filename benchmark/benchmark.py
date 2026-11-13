"""가짜 모델 서버에서 동기 체인과 비동기 그래프의 응답 시간과 처리량을 잰다.

두 경로의 작업량은 다르다. 시험용 리뷰를 먼저 100개 만들고, 경로마다 리뷰별로 한 번씩 요청한다.
실행 방법과 지표는 README를 참고한다.
"""

import asyncio
import statistics
import time

import httpx

# Windows에서 localhost는 IPv6(::1)를 먼저 시도하느라 연결마다 지연이 생길 수 있어서 127.0.0.1을 쓴다.
BASE_URL = "http://127.0.0.1:8000"

# (이름, 리뷰 번호 뒤에 붙일 경로)
TARGETS = [
    ("1) reply/sync", "reply/sync"),
    ("2) reply", "reply"),
]
REVIEW_COUNT = 100
# 요청을 나눠 보낼 클라이언트 수. 클라이언트 하나로 100개를 보내면 측정 도구 쪽이 먼저 느려진다(chap03).
CLIENT_COUNT = 5


async def create_benchmark_reviews(client: httpx.AsyncClient) -> list[int]:
    ids = []
    for index in range(REVIEW_COUNT):
        response = await client.post("/api/reviews", json={
            "customerName": f"측정 손님 {index + 1}",
            "rating": 5,
            "content": "열무김치라떼가 고소했고 직원분들이 친절했어요.",
        })
        response.raise_for_status()
        ids.append(response.json()["reviewId"])
    return ids


async def send_one(client: httpx.AsyncClient, path: str) -> tuple[float, bool]:
    """요청 하나를 보내고 (걸린 시간(초), 성공 여부)를 반환한다."""

    start = time.perf_counter()
    try:
        response = await client.post(path)
        ok = response.status_code == 200
    except httpx.HTTPError:
        ok = False
    return time.perf_counter() - start, ok


async def measure(clients: list[httpx.AsyncClient], name: str, suffix: str, ids: list[int]) -> None:
    """리뷰마다 동시에 한 요청을 보내고 지표를 출력한다."""

    start = time.perf_counter()
    results = await asyncio.gather(*(
        send_one(clients[index % len(clients)], f"/api/reviews/{review_id}/{suffix}")
        for index, review_id in enumerate(ids)
    ))

    total = time.perf_counter() - start
    count = len(ids)
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
        health = await clients[0].get("/health")
        if health.status_code != 200 or health.json().get("model") != "fake-llm":
            print("FAKE_LLM=true로 실행한 로컬 서버에서 측정해 주세요.")
            return

        ids = await create_benchmark_reviews(clients[0])
        for name, suffix in TARGETS:
            await measure(clients, name, suffix, ids)
    finally:
        for client in clients:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
