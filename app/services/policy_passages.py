"""Select an answering sentence from the actual hybrid candidates."""

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from math import log

from langchain_core.documents import Document

from app.services.policy_topics import PolicyEvidence, article_sentences


DOCUMENT_TOPIC = "document_question"
# Terms used to distinguish nearby policy topics. These also apply to unseen articles.
ALIASES = tuple((re.compile(pattern), term) for pattern, term in (
    (r"엘리베이터|리프트", "승강기"),
    (r"(?:아기|어린이)(?=\s*(?:용|의자|식기|컵|수저))|유아용", "유아"),
    (r"컵[·ㆍ와과]?\s*수저|어린이용\s*(?:컵|수저)", "식기"),
    (r"첫\s*번째|1차", "첫"),
    (r"둘이\s*앉을\s*자리|일반\s*(?:좌석|자리)(?:\s*[2-9]\s*명)?\s*예약|소규모\s*예약", "일반 좌석 예약"),
    (r"(?<!\d)[2-9]\s*(?:인|명)\s*예약|두\s*명\s*예약", "일반 좌석 예약"),
    (r"예약할|예약하는", "예약"),
    (r"다시\s*부르|더\s*불러|재호출", "호출"),
    (r"잃어버린\s*(?:물건|지갑|가방|우산)?|놓고\s*온\s*(?:물건|지갑|가방|우산)|두고\s*간\s*(?:물건|지갑|가방|우산)|습득물|발견한\s*물건", "분실물"),
    (r"택배로\s*보내|우편(?:으로)?\s*보내|택배\s*발송", "택배 발송"),
    (r"일행이\s*늦|예약\s*시(?:각|간)보다\s*늦|늦을\s*(?:경우|때)|방문\s*시각\s*지연|예약\s*지각", "예약 지연"),
    (r"자리(?:를)?\s*(?:얼마나\s*)?잡아\s*줘", "자리 유지"),
    (r"카드\s*승인[^.?!]{0,12}두\s*(?:줄|건)|이중\s*결제|두\s*번\s*(?:카드\s*)?결제", "카드 중복 결제"),
    (r"대신\s*(?:수령|받|찾아|가져가)|(?:다른\s*사람|제가|가족|친구)(?:이|가)?\s*대신[^.?!]{0,12}(?:받으러|찾으러)|대리\s*수령", "대리수령"),
    (r"주문\s*번호", "주문번호"),
    (r"두\s*(?:개\s*)?(?:집|곳|군데)|(?:여러|다른)\s*(?:배송지|배달지|주소)|주소[^.?!]{0,12}두\s*개", "복수주소"),
    (r"사진[^.?!]{0,21}(?:못\s*올려|안\s*올라|업로드[^.?!]{0,7}실패|첨부[^.?!]{0,8}(?:오류|실패|안\s*돼))", "사진 첨부 오류"),
    (r"업로드|첨부", "첨부"),
    (r"알림(?:이|을|은)?\s*(?:안\s*왔|오지\s*않|못\s*받)|알림\s*미수신", "알림 미수신"),
    (r"됐는지\s*어디서\s*봐|확정\s*여부", "확정 여부"),
    (r"적립분부터\s*빠지|포인트[^.?!]{0,12}먼저\s*빠지|차감", "차감"),
    (r"잔액을\s*보내|포인트를\s*보내|포인트\s*양도", "포인트 양도"),
    (r"반려견|강아지", "반려동물"),
    (r"충전\s*케이블|충전기|충전\s*어댑터", "충전 케이블"),
    (r"빌릴|빌려|빌리", "대여"),
))
# Focus terms in the question must be present in a candidate's title or selected sentence.
FOCUS_TERMS = frozenset((
    "승강기", "휠체어", "유아", "식기", "삼각대", "택배", "분실물", "보관", "호출", "지연",
    "중복", "확인증", "대리수령", "복수주소", "첨부", "차감", "양도", "미수신", "충전", "케이블", "제출",
))
STOP_WORDS = frozenset((
    "어느", "어디", "언제", "얼마", "얼마나", "어떤", "어떻게", "궁금", "경우", "다시", "가능",
    "오늘", "어제", "지금", "제가", "제게", "받을", "되나요", "되어", "봐요", "있나요", "해요",
    "하나요", "되죠", "주세요", "있는", "알려", "며칠", "동안", "걸리는", "업무", "계속", "얼마예요",
    "우리", "뒤까지", "넣을", "받는", "싶은데", "가는데", "가면", "는데", "받아", "하나", "가족", "직접", "가지", "못하면",
))
WORD = re.compile(r"[가-힣a-z0-9]+")
ENDING = re.compile(r"(?:에서는|에서|에게|한테|까지|으로|이라면|과는|에도|부터|으로는|[은는이가을를과와도에의로랑])$")
VERB_ENDING = re.compile(r"(?:해\s*줘요|해요|하나요|했어요|합니다|한다|인가요|있나요|있어요|되나요|돼요|됐나요|할까요|주시나요)$")
ANSWER_FORMS = tuple((re.compile(question), re.compile(answer)) for question, answer in (
    (r"몇\s*(?:개|명|석|자리|장)", r"(?:\d+|[한두세네]|다섯|여섯|일곱|여덟|아홉|열)\s*(?:개|명|석|자리|장)"),
    (r"며칠|몇\s*일", r"\d+\s*일"),
    (r"언제까지", r"(?:(?:오전|오후)\s*\d+\s*시|\d+\s*(?:시|일|분|시간))[^.?!]{0,12}(?:까지|안에)|전날|당일"),
    (r"몇\s*(?:분|시간)|얼마나[^.?!]{0,15}(?:잡아|유지)|유지\s*시간", r"\d+\s*(?:분|시간)"),
    (r"몇\s*시(?!간)", r"\d+\s*시"),
    (r"퍼센트|비율|몇\s*%", r"\d+\s*%"),
    (r"최소\s*주문\s*금액|얼마(?:예요|인가요|를)|요금은", r"[\d,]+\s*원"),
    (r"주소[^.?!]{0,16}(?:두\s*개|한\s*개|몇\s*개)|주문\s*하나[^.?!]{0,25}두\s*개", r"하나의|한\s*곳|주소\s*한\s*개"),
    (r"어디(?:에|서)[^.?!]{0,9}(?:접수|봐|확인)|어떤\s*채널", r"앱|매장|전화|화면|카운터|안내\s*데스크"),
    (r"됐는지\s*어디서\s*봐|확정\s*여부[^.?!]{0,12}(?:어디|확인)", r"확정\s*여부"),
    (r"어느\s*(?:구역|승강기|층|출입구)", r"승강기|출입구|\d+\s*층"),
))
COMPOUND_QUESTIONS = re.compile(r"(?:시간|기간|기한|방법|조건|요금)(?:과|와|이랑|랑)\s+(?=[가-힣])")
CONTEXT_QUESTION = re.compile(r"^(?:그럼\s*)?(?:어디[에서]*|언제|어떻게|얼마나|몇\s*[시분일개명])")
MIN_PASSAGE_MATCH = 0.43
AMBIGUITY_MARGIN = 0.035


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    for pattern, word in ALIASES:
        text = pattern.sub(f" {word} ", text)
    return text


def terms(text: str) -> frozenset[str]:
    result = set()
    for raw in WORD.findall(text):
        word = VERB_ENDING.sub("", raw)
        stem = ENDING.sub("", word)
        if len(stem) >= 2:
            word = stem
        if (len(word) >= 2 or word == "앱") and word not in STOP_WORDS and not word.isdigit():
            result.add(word)
    return frozenset(result)


def relevant_words(query: frozenset[str], source: frozenset[str]) -> frozenset[str]:
    return frozenset(word for word in query if word in source or (
        len(word) >= 2 and any(
            (len(word) >= 3 and other.startswith(word))
            or (len(other) >= 2 and len(word) >= 3 and word.startswith(other))
            for other in source
        )
    ))


def split_questions(question: str) -> tuple[str, ...]:
    """Separate an explicit pair of requested fields while keeping the original for retrieval."""
    parts = []
    start = 0
    for joint in COMPOUND_QUESTIONS.finditer(question):
        first = question[start:joint.end()].rstrip()
        first = re.sub(r"(?:이랑|랑|과|와)$", "", first)
        parts.append(first)
        start = joint.end()
    return (*parts, question[start:].strip())


def requested_forms(question: str) -> list[re.Pattern]:
    return [answer for ask, answer in ANSWER_FORMS if ask.search(question)]


class PolicyPassages:
    def __init__(self, articles: Sequence[Document]):
        self._headings = {}
        self._passages = {}
        frequencies: Counter[str] = Counter()
        for article in articles:
            title = article.metadata.get("article", "")
            heading = terms(normalized(title))
            passages = tuple((sentence, terms(normalized(sentence))) for sentence in article_sentences(article))
            self._headings[article.id] = heading
            self._passages[article.id] = passages
            frequencies.update(heading.union(*(words for _, words in passages)))
        self._weights = {word: 1 + log((len(articles) + 1) / (count + 1)) for word, count in frequencies.items()}

    def select(
        self, question: str, candidates: list[Document], *, context: str = "",
    ) -> tuple[Document, PolicyEvidence] | None:
        query = terms(normalized(question))
        if context and CONTEXT_QUESTION.search(question):
            query |= terms(normalized(context)) & FOCUS_TERMS
        weights = {word: self._weights.get(word, 1.0) for word in query}
        total = sum(weights.values())
        if not total:
            return None
        focus = query & FOCUS_TERMS
        answer_forms = requested_forms(question)
        matches: list[tuple[float, tuple[str, ...], Document]] = []
        for position, article in enumerate(candidates):
            heading = self._headings.get(article.id, frozenset())
            passages = self._passages.get(article.id, ())
            best: tuple[float, tuple[str, ...]] | None = None
            for index, (sentence, words) in enumerate(passages):
                if any(not form.search(sentence) for form in answer_forms):
                    continue
                scope = heading | words
                excerpt = (sentence,)
                if index and answer_forms and not (focus & words):
                    previous, previous_words = passages[index - 1]
                    if relevant_words(query, previous_words) - relevant_words(query, scope):
                        scope |= previous_words
                        excerpt = (previous, sentence)
                if focus - scope:
                    continue
                matched = relevant_words(query, words)
                if not matched and not answer_forms:
                    continue
                direct = sum(weights[word] for word in matched) / total
                surrounding = sum(weights[word] for word in relevant_words(query, scope)) / total
                if not answer_forms and focus and not (focus & words):
                    continue
                if not matched and (not index or surrounding < 0.75):
                    continue
                score = direct + (0.55 if answer_forms else 0.23) * surrounding + 0.10 / (position + 1)
                if score >= MIN_PASSAGE_MATCH and (best is None or score > best[0]):
                    best = (score, excerpt)
            if best:
                matches.append((*best, article))
        matches.sort(key=lambda match: match[0], reverse=True)
        if not matches or (len(matches) > 1 and matches[0][0] - matches[1][0] < AMBIGUITY_MARGIN):
            return None
        _, excerpt, article = matches[0]
        return article, PolicyEvidence(DOCUMENT_TOPIC, article.id, excerpt, None, False)

    def matches_scope(self, question: str, evidence: PolicyEvidence, *, check_form: bool) -> bool:
        """A broad known rule does not cover a narrower subject mentioned in the question."""
        focus = terms(normalized(question)) & FOCUS_TERMS
        heading = self._headings.get(evidence.article_id, frozenset())
        source = terms(normalized(" ".join(evidence.excerpts)))
        if not focus.issubset(heading | source):
            return False
        return not check_form or all(
            any(form.search(sentence) for sentence in evidence.excerpts)
            for form in requested_forms(question)
        )
