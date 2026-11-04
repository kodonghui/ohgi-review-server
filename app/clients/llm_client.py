"""
LLM 호출 클라이언트 (Spring Boot에서 RestClient로 OpenAI를 호출하던 클래스의 역할)

chap02의 llm_client.py를 가져와 두 가지를 바꿨다.
1. 두 번째 매개변수 이름을 question에서 prompt로 바꿨다. 답글 서비스가 리뷰로 만든 프롬프트를 넘기기 때문이다.
2. 응답 문장을 꺼내기 전에 응답이 끝까지 만들어졌는지 확인한다(read_output_text).
   출력 토큰 상한에 걸려 중간에 끊긴 답글을 그대로 돌려주면 사장님이 잘린 문장을 그대로 올릴 수 있어서, 502로 알린다.

이 클래스는 동기 클라이언트(openai.OpenAI)와 비동기 클라이언트(openai.AsyncOpenAI)를 함께 가지고, 같은 일을 하는 메서드를 두 개 제공한다.
- ask()       : 동기. OpenAI의 응답이 올 때까지 이 메서드를 실행한 스레드가 멈춰서 기다린다
- ask_async() : 비동기. 응답을 기다리는 동안 이벤트 루프가 다른 일을 할 수 있다

키가 없거나 FAKE_LLM=true면 OpenAI를 호출하지 않는 FakeLlmClient를 대신 사용한다(main.py의 lifespan).
두 클래스의 메서드 이름과 매개변수가 같아서, 답글 서비스는 어느 쪽을 전달받든 코드를 바꾸지 않아도 된다.
"""

import asyncio
import time

import openai

from app.core.config import Settings
from app.core.exceptions import AiRequestError


class OpenAiLlmClient:

    def __init__(self, settings: Settings):
        # 전달받은 설정 객체(Settings)를 보관한다
        self.settings = settings
        # 응답 DTO에 담을 모델 이름
        self.model = settings.openai_model

        # lifespan을 거치지 않고 이 클래스를 직접 만들었는데 키가 없으면(테스트 등) 원인을 알려 주며 멈춘다.
        if settings.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY가 없어 OpenAI 클라이언트를 만들 수 없습니다.")

        # 동기 클라이언트와 비동기 클라이언트를 같은 설정으로 만든다.
        # - api_key : SecretStr에서 실제 키 문자열을 꺼내 전달한다
        # - timeout : 응답을 기다리는 최대 시간(초)
        # - max_retries : 연결 실패나 서버 오류일 때 다시 시도할 횟수
        api_key = settings.openai_api_key.get_secret_value()

        self._client = openai.OpenAI(
            api_key=api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=1,
        )

        self._async_client = openai.AsyncOpenAI(
            api_key=api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=1,
        )

    # 1. 동기 호출 : 응답이 올 때까지 스레드가 멈춰 있다.
    def ask(self, instructions: str, prompt: str) -> str:
        """
        지시문과 프롬프트를 OpenAI에 보내고 응답 문장을 반환한다. (동기)

        Args:
            instructions (str): 지시문(시스템 프롬프트). LLM이 어떤 역할로, 어떻게 응답할지 정한다.
            prompt (str): LLM에 보낼 입력(사용자 메시지). 답글 서비스가 저장된 리뷰로 만든다.

        Returns:
            str: LLM이 만든 응답 문장

        Raises:
            AiRequestError: OpenAI 호출이 실패했거나, 응답이 끊겼거나 비어 있을 때
        """

        start = time.perf_counter()

        try:
            # OpenAI Responses API 호출 (POST https://api.openai.com/v1/responses)
            response = self._client.responses.create(
                model=self.settings.openai_model,
                instructions=instructions,
                # 문자열 하나를 전달하면 사용자(user)가 보낸 메시지 하나로 처리된다.
                input=prompt,
                max_output_tokens=self.settings.openai_max_output_tokens,
                reasoning={"effort": self.settings.openai_reasoning_effort},
                # 리뷰 내용을 OpenAI 서버에 저장하지 않는다
                store=False,
            )
        # openai 라이브러리의 예외(401, 429, 시간 초과 등)를 우리 예외로 바꿔서 다시 던진다.
        except openai.OpenAIError as e:
            raise to_ai_request_error(e) from e

        print_usage("ask", response, start)

        return read_output_text(response)

    # 2. 비동기 호출 : ask()와 같은 요청을 보내고, 다른 점은 async def, AsyncOpenAI, await 세 가지다.
    # await 지점에서 이 코루틴은 멈추고, 응답이 오기 전까지 이벤트 루프는 다른 요청을 처리한다.
    async def ask_async(self, instructions: str, prompt: str) -> str:
        """
        지시문과 프롬프트를 OpenAI에 보내고 응답 문장을 반환한다. (비동기)

        Args:
            instructions (str): 지시문(시스템 프롬프트)
            prompt (str): LLM에 보낼 입력(사용자 메시지)

        Returns:
            str: LLM이 만든 응답 문장

        Raises:
            AiRequestError: OpenAI 호출이 실패했거나, 응답이 끊겼거나 비어 있을 때
        """

        start = time.perf_counter()

        try:
            response = await self._async_client.responses.create(
                model=self.settings.openai_model,
                instructions=instructions,
                input=prompt,
                max_output_tokens=self.settings.openai_max_output_tokens,
                reasoning={"effort": self.settings.openai_reasoning_effort},
                store=False,
            )
        except openai.OpenAIError as e:
            raise to_ai_request_error(e) from e

        print_usage("ask_async", response, start)

        return read_output_text(response)

    # 서버가 종료될 때(lifespan) 호출해서 OpenAI 서버와의 연결을 닫는다.
    # 비동기 클라이언트의 close()가 async def라서 이 메서드도 async def다.
    async def close(self) -> None:
        self._client.close()
        await self._async_client.close()


def read_output_text(response) -> str:
    """
    응답에서 문장을 꺼낸다. 답글로 쓸 수 없는 응답이면 AiRequestError를 던진다.

    - status가 "incomplete"면 응답이 중간에 끊긴 것이다. 가장 흔한 이유는 출력 토큰 상한(max_output_tokens)이다.
    - 문장이 비어 있으면 사장님이 쓸 답글이 없다.

    Args:
        response: OpenAI Responses API의 응답 객체

    Returns:
        str: 앞뒤 공백을 지운 응답 문장

    Raises:
        AiRequestError: 응답이 끊겼거나 비어 있을 때 (예외 처리기가 502로 응답한다)
    """

    if response.status == "incomplete":
        details = response.incomplete_details
        reason = details.reason if details is not None else None
        if reason == "max_output_tokens":
            raise AiRequestError("OpenAI 응답이 출력 토큰 상한에 걸려 중간에 끊겼습니다. OPENAI_MAX_OUTPUT_TOKENS를 늘리세요.")
        raise AiRequestError(f"OpenAI 응답이 완성되지 않았습니다. (이유: {reason})")

    text = response.output_text.strip()
    if not text:
        raise AiRequestError("OpenAI 응답에 답글 문장이 없습니다.")

    return text


def print_usage(method: str, response, start: float) -> None:
    """걸린 시간, 응답 상태, 토큰 사용량을 콘솔에 찍는다. 비용과 끊긴 응답을 확인할 때 사용한다."""

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    usage = response.usage

    if usage is not None:
        print(f"[OpenAiLlmClient.{method}] model={response.model}, status={response.status}, "
              f"입력 토큰={usage.input_tokens}, 출력 토큰={usage.output_tokens}, {elapsed_ms}ms")


def to_ai_request_error(e: openai.OpenAIError) -> AiRequestError:
    """
    openai 라이브러리의 예외를 우리 예외로 바꾼다. (Spring Boot에서 HTTP 상태 코드를 보고 우리 예외로 바꿨던 일)

    Args:
        e (openai.OpenAIError): OpenAI 호출 중 발생한 예외

    Returns:
        AiRequestError: 대신 던질 예외. 던지는 것은 호출한 쪽(raise ... from e)이다.
    """

    # 401 : API 키가 틀렸거나, 삭제되었거나, 빈 문자열이다.
    if isinstance(e, openai.AuthenticationError):
        return AiRequestError("OpenAI API 키가 올바르지 않습니다. OPENAI_API_KEY를 확인하세요.")

    # 429 : 짧은 시간에 요청을 너무 많이 보냈거나, 계정의 충전 잔액이 바닥났다.
    if isinstance(e, openai.RateLimitError):
        return AiRequestError("OpenAI 요청 한도를 넘었거나 충전 잔액이 부족합니다.")

    # 시간 초과 : APITimeoutError는 APIConnectionError의 자식 클래스라서 먼저 검사한다.
    if isinstance(e, openai.APITimeoutError):
        return AiRequestError("OpenAI 응답이 제한 시간 안에 오지 않았습니다.")

    # 연결 실패 : 인터넷 연결이 끊겼거나, 방화벽·프록시가 막았다. (응답을 받지 못했으므로 HTTP 상태 코드가 없다)
    if isinstance(e, openai.APIConnectionError):
        return AiRequestError("OpenAI 서버에 연결할 수 없습니다.")

    # 그 밖의 오류 : 400(요청 형식 오류, 없는 모델 이름 등), 403(권한 없음), 500번대(OpenAI 서버 오류) 등
    return AiRequestError(f"OpenAI 호출 실패 : {e}")


class FakeLlmClient:
    """
    OpenAI를 호출하지 않는 가짜 클라이언트. 키가 없거나 FAKE_LLM=true일 때 사용한다.
    LLM이 응답하는 데 걸리는 시간만 흉내 내고, 받은 프롬프트를 그대로 담아 돌려준다.
    그래서 가짜 응답만 보고도 저장된 리뷰가 프롬프트에 들어갔는지 확인할 수 있다.
    """

    def __init__(self, settings: Settings):
        self.delay_seconds = settings.fake_llm_delay_seconds
        self.model = "fake-llm"

    # time.sleep : 이 스레드를 멈춰 세운다. 동기 클라이언트가 응답을 기다리는 것과 같은 상황을 만든다.
    def ask(self, instructions: str, prompt: str) -> str:
        time.sleep(self.delay_seconds)
        return f"(가짜 응답) {self.delay_seconds}초 기다렸습니다. 받은 프롬프트 : {prompt}"

    # await asyncio.sleep : 이 코루틴만 멈춘다. 기다리는 동안 이벤트 루프는 다른 요청을 처리한다.
    async def ask_async(self, instructions: str, prompt: str) -> str:
        await asyncio.sleep(self.delay_seconds)
        return f"(가짜 응답) {self.delay_seconds}초 기다렸습니다. 받은 프롬프트 : {prompt}"

    async def close(self) -> None:
        pass


# 두 클라이언트 중 하나라는 뜻의 타입 이름. 의존성 함수와 답글 서비스의 타입 힌트에 사용한다.
LlmClient = OpenAiLlmClient | FakeLlmClient
