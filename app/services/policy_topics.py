"""규정 문의와 이를 직접 뒷받침하는 조문 문장을 연결한다."""

import re
from dataclasses import dataclass

from langchain_core.documents import Document


QUALITY_ARTICLE = "제12조"
DELIVERY_ARTICLE = "제17조"
QUALITY_ISSUE = r"이물질|오배송|다른\s*(?:메뉴|음식)|메뉴가\s*잘못|주문한\s*것과\s*다른"
DELIVERY_DAMAGE = r"(?:용기|포장|배달)[^.!?\n]{0,16}(?:파손|깨졌|깨져|터[졌지])|쏟아[졌져]"
INCIDENT_DENIAL = re.compile(
    r"\s*(?:[은는이가도]|된\s*(?:건|게|것)(?:은|이)?)?\s*"
    r"(?:없|아니|아닙|안\s*(?:왔|나왔)|(?:오|나|되)지\s*않)"
)
TASTE_COMPLAINT = r"(?:맛|입맛)[^.!?\n]{0,24}(?:안\s*맞|맞지\s*않|별로|없)"
ALCOHOL = r"맥주|주류|술|와인|라거|소주|막걸리"
DELIVERY = r"배달|배송|집으로|가져다"
POLICY_QUESTION = re.compile(
    r"[?？]|궁금|문의|되나요|있나요|없나요|알려\s*(?:주세요|주실|줘)|"
    r"환불\s*(?:해|할|요청|받고)|(?:재배달|보상|교환)\s*(?:해|받|되|주세요)|"
    r"언제|몇\s*시|(?:얼마|몇\s*점)[^.!?]{0,12}(?:예요|인가|받|돼|쌓)|"
    r"영업\s*시간|배달비|브레이크\s*타임|예약\s*가능|포장\s*가능"
)
PICKUP_SUBJECT = r"(?:포장|테이크\s*아웃|주문|음식|분실물|지갑|가방|우산|물건)"
PICKUP_COMPLETED = (
    r"(?:받아\s*갔|받았|가져\s*갔|찾아\s*갔|찾았|"
    r"(?:픽업|수령|인계)(?:[이가을를])?\s*(?:완료\s*)?(?:했|됐|되었))"
)
STATUS_QUESTION = r"(?:나요|는지|습니까|어요(?=\s*[?？])|[?？])"
STAFF_ONLY = re.compile(
    r"알레르기|칼로리|대여료|개인\s*전화번호|직원[^.!?]{0,20}(?:개인\s*연락처|성함)|"
    r"실제로\s*비어|(?:좌석|자리)[^.!?]{0,24}(?:비어\s*있|남아\s*있)|"
    r"(?:환불|주문)[^.!?]{0,26}(?:이미\s*승인|승인\s*됐|입금\s*됐|처리\s*완료\s*됐)|"
    r"환불[^.!?\n]{0,35}(?:들어\s*왔|돌아\s*왔|(?:입금|승인|처리)[이가]?\s*(?:됐|되었|완료))|"
    r"(?:환불|취소|입금|승인)(?:금)?[이가은는]?\s*(?:이미\s*)?(?:됐|되었)(?:는지|나요|어요|습니까|[?？])|"
    r"(?:주차\s*지원|차량?\s*번호)[^.!?\n]{0,35}등록\s*(?:됐|되었|되어\s*있)|"
    r"(?:분실물|지갑|가방|우산|휴대폰)[^.!?\n]{0,35}(?:들어와\s*있|습득\s*됐|접수\s*됐|보관\s*중)|"
    rf"{PICKUP_SUBJECT}[^.!?\n]{{0,42}}{PICKUP_COMPLETED}{STATUS_QUESTION}|"
    r"(?:케이크|외부\s*음식|음식물)[^.!?\n]{0,18}(?:가져가|가져와|반입)[^.!?\n]{0,20}(?:되나|돼요|될까|가능|허용)|"
    r"대기\s*(?:순서|순번|번호)[^.!?\n]{0,18}(?:몇\s*번|어디까지|제\s*차례)|"
    r"관리자[^.!?\n]{0,12}(?:암호|비밀번호|패스워드)|"
    r"쿠폰[^.!?]{0,18}(?:발급|보상|주세요|주실|줘|달라|준다고|드린다고|안\s*됐)"
)
FOLLOWUP_QUESTION = re.compile(r"(?:각각\s*)?(?:사진을?\s*언제까지\s*보내나요[?？]?|환불해\s*주세요[.!]?)")


def article_sentences(article: Document) -> tuple[str, ...]:
    lines = article.page_content.splitlines()
    if len(lines) > 1 and lines[0].startswith(str(article.id)):
        lines = lines[1:]
    return tuple(
        sentence.strip().removeprefix("- ")
        for sentence in re.split(r"(?<=[.!?。])\s+|\n+", "\n".join(lines))
        if sentence.strip()
    )


@dataclass(frozen=True)
class PolicyEvidence:
    topic: str
    article_id: str
    excerpts: tuple[str, ...]
    hours: str | None
    followup: bool

    def required_notice(self) -> str | None:
        if self.topic == "quality_incident":
            return f"다른 메뉴나 이물질은 수령 후 {self.hours}시간 이내에 사진과 함께 알려 주세요."
        if self.topic == "delivery_damage":
            return f"용기 파손이나 내용물 유출은 수령 후 {self.hours}시간 이내에 사진과 함께 알려 주세요."
        if self.topic == "alcohol_delivery":
            return "주류는 배달하지 않습니다."
        return None


@dataclass(frozen=True)
class PolicyRule:
    topic: str
    article_id: str
    query: str
    question_pattern: re.Pattern
    source_pattern: re.Pattern
    direct: bool = False
    followup: bool = False
    deadline: bool = False

    def matches(self, text: str) -> bool:
        if not self.deadline:
            return bool(self.question_pattern.search(text))
        return any(
            not INCIDENT_DENIAL.match(text, match.end()) for match in self.question_pattern.finditer(text)
        )

    def evidence(self, article: Document) -> PolicyEvidence | None:
        excerpts = tuple(clause for clause in article_sentences(article) if self.source_pattern.search(clause))
        if not excerpts:
            return None
        hours = re.search(r"(\d+)\s*시간", excerpts[0]) if self.deadline else None
        if self.deadline and hours is None:
            return None
        return PolicyEvidence(self.topic, self.article_id, excerpts,
                              hours.group(1) if hours else None, self.followup)


def rule(topic, article, query, question, source, **options):
    return PolicyRule(topic, f"제{article}조", query, re.compile(question), re.compile(source), **options)


RULES = (
    rule("quality_incident", 12, "오배송 이물질 수령 접수 기한 사진", QUALITY_ISSUE,
         r"(?:이물질|다른\s*메뉴)[^.]{0,105}\d+\s*시간[^.]{0,32}사진",
         direct=True, followup=True, deadline=True),
    rule("delivery_damage", 17, "배달 용기 파손 내용물 유출 접수 기한 사진", DELIVERY_DAMAGE,
         r"(?:용기.{0,12}파손|쏟아지)[^.]{0,105}\d+\s*시간[^.]{0,32}사진",
         direct=True, followup=True, deadline=True),
    rule("taste_refund", 12, "맛 불만 교환 환불", rf"(?=.*(?:{TASTE_COMPLAINT}))(?=.*환불)",
         r"맛이[^.]*환불을\s*하지\s*않", direct=True, followup=True),
    rule("alcohol_delivery", 18, "주류 배달 맥주 매장 판매", rf"(?=.*(?:{ALCOHOL}))(?=.*(?:{DELIVERY}))",
         r"^주류는\s*배달하지\s*않는다[.。]?$"),
    rule("delivery_hours", 15, "배달 주문 시간 마감 배달 예상 시간",
         rf"(?=.*(?:{DELIVERY}))(?=.*(?:마감|오래|몇\s*시|걸려|예상\s*시간|밤\s*\d))",
         r"배달 주문은|배달 예상 시간"),
    rule("delivery_minimum", 16, "배달 최소 주문 금액 배달비",
         rf"배달비|(?=.*(?:{DELIVERY}))(?=.*(?:\d[\d,]*\s*원|최소\s*주문))",
         r"배달은.*주문|배달비"),
    rule("store_hours", 1, "영업시간 브레이크타임 마지막 매장 주문",
         r"영업\s*시간|개점|문을?\s*(?:열|닫)|브레이크\s*타임|마지막\s*(?:매장\s*)?주문|"
         r"평일[^.?!]{0,25}(?:매장|식사)|매장\s*주문[^.?!]{0,20}마감",
         r"매장은.*영업|마지막 주문|브레이크타임|이 시간에는"),
    rule("closed_days", 2, "휴무 월요일 공휴일 다음 날 화요일",
         r"휴무|쉬는\s*날|월[요욜]|공휴일|빨간\s*날", r"정기 휴무일|월요일이 공휴일|설날과 추석"),
    rule("animals", 3, "반려동물 매장 이용 장애인 보조견",
         r"강아지|반려동물|애완동물|고양이|보조견", r"반려동물|장애인 보조견"),
    rule("parking", 4, "주차 무료 결제 금액 초과 요금",
         r"주차|차도\s*가져|차를\s*(?:가져|세워)", r"주차장|무료 주차|주차 요금"),
    rule("wifi", 5, "와이파이 비밀번호 영수증 콘센트",
         r"와이파이|wi-fi|콘센트", r"와이파이|비밀번호|콘센트"),
    rule("ordering", 6, "매장 태블릿 직원 주문 전화 주문 배달 포장",
         r"(?:전화|태블릿|점원|직원|QR)[^.?!]{0,28}주문|포장\s*가능", r"태블릿|직원에게|포장과 배달|전화 주문"),
    rule("restocking", 7, "품절 재료 재입고 날짜 주문 불가",
         r"재입고|품절|계절\s*메뉴|주문\s*불가", r"주문 불가|재입고"),
    rule("payments", 8, "결제 수단 상품권 모바일 상품권",
         r"상품권|결제\s*수단|(?:간편결제|카드로|현금으로)[^.?!]{0,12}결제", r"상품권|결제 수단|현금으로"),
    rule("receipt", 9, "현금영수증 발급 결제 후 매장 요청",
         r"현금\s*영수증|영수증[^.?!]{0,16}(?:발급|받)", r"현금영수증|결제한 날로부터"),
    rule("reusable_box", 10, "포장 다회용기 반납 할인 보증금",
         r"다회용기|(?:포장\s*(?:용기|그릇)|그릇)[^.?!]{0,22}(?:돌려|반납|보증금|일주일)",
         r"다회용기|용기는.*돌려준다"),
    rule("cancel_order", 11, "조리 시작 전 주문 취소 전액 취소",
         r"(?:조리|주문\s*상태)[^.?!]{0,38}취소|주문을?\s*취소", r"조리가|조리 중이라도", followup=True),
    rule("refund_timing", 13, "카드 결제 취소 환불 처리 기간 영업일",
         r"(?:카드|간편결제|현금)[^.?!]{0,24}(?:취소|환불)(?:[^.?!]{0,24}(?:걸|돌아|며칠|언제))?",
         r"카드 결제 취소|간편결제는|현금으로 결제|포인트로 결제", followup=True),
    rule("partial_refund", 14, "여러 메뉴 중 일부 문제 부분 환불",
         r"부분\s*환불|(?:하나|일부|한\s*개)만[^.?!]{0,36}(?:값|금액|환불|돌려)", r"일부에만|부분 환불",
         followup=True),
    rule("origin", 21, "메뉴 원산지 국내산 수입산",
         r"원산지|(?:새우|쇠고기|돼지고기|닭고기|오징어|갈치|우럭)[^.?!]{0,20}(?:어디|어느\s*나라|산인가)",
         r"국내산|베트남산|원양산"),
    rule("spice", 22, "맵기 조절 순한맛 어린이",
         r"맵기|순한\s*맛|매운\s*맛|초등학생[^.?!]{0,18}맵", r"순한맛|매운맛"),
    rule("membership_tier", 24, "멤버십 새싹 열매 나무 등급 기준 결제 금액",
         r"(?:등급|열매|새싹|나무)[^.?!]{0,28}(?:되려|써야|산정|기준|올라)", r"등급은 매월|최근 3개월 결제"),
    rule("points", 26, "포인트 사용 조건 유효기간",
         r"(?:포인트|적립금)[^.?!]{0,32}(?:쓸|사용|유효|만료)|모은\s*지", r"포인트는|포인트로는|포인트의|기한이 지난"),
    rule("coupon_use", 27, "쿠폰 장수 중복 사용 포인트 함께",
         r"쿠폰[^.?!]{0,32}(?:두\s*장|두\s*개|동시|같이|함께|중복)", r"쿠폰은 한 번|여러 쿠폰|함께 사용할"),
    rule("group_booking", 30, "단체 예약 인원 10명 예약금 인당 5,000원",
         r"(?:단체|\d+\s*명|열두\s*명|열\s*명)[^.?!]{0,18}예약", r"10명 이상|단체 예약은"),
    rule("group_cancellation", 31, "단체 예약 취소 전날 예약금 환불 비율",
         r"(?:예약|방문|전날|하루\s*전)[^.?!]{0,23}취소", r"방문일|예약금을"),
    rule("review_points", 33, "앱 주문 사진 리뷰 적립 포인트",
         r"(?:사진[^.?!]{0,16}(?:리뷰|후기)|리뷰\s*포인트)[^.?!]{0,18}(?:포인트|몇\s*점|적립|궁금|쌓)|"
         r"사진[^.?!]{0,16}(?:리뷰|후기).*쌓", r"앱에서 주문한 손님은|주문 1건"),
    rule("privacy", 34, "멤버십 탈퇴 개인정보 생일 보관",
         r"탈퇴|개인정보[^.?!]{0,19}(?:보관|지워|없어|삭제)", r"탈퇴하면|멤버십 가입 때"),
    rule("contact", 35, "일대일 문의 답변 시간",
         r"일대일\s*문의|1\s*:\s*1\s*문의|문의[^.?!]{0,18}언제.*답변", r"문의는|하루 안에"),
)


def route_questions(text: str) -> tuple[tuple[tuple[PolicyRule, str], ...], tuple[str, ...], str]:
    """Return registered rules and the questions that need document-based lookup."""
    if STAFF_ONLY.search(text):
        return (), (), "staff_only"

    topics: dict[str, tuple[PolicyRule, str]] = {}
    unresolved = []
    for part in re.split(r"(?<=[?!。？])\s*|\.(?=\s|$)|\n+", text):
        part = part.strip()
        if not part:
            continue
        inquiry = POLICY_QUESTION.search(part)
        found = [item for item in RULES if (inquiry or item.deadline) and item.matches(part)]
        if any(item.topic == "closed_days" for item in found) and not re.search(
            r"영업\s*시간|마지막|브레이크|몇\s*시", part
        ):
            found = [item for item in found if item.topic != "store_hours"]
        if any(item.topic == "group_cancellation" for item in found) and not re.search(r"인당|먼저\s*내", part):
            found = [item for item in found if item.topic != "group_booking"]
        if any(item.topic == "refund_timing" for item in found) and not re.search(r"상품권|결제\s*수단", part):
            found = [item for item in found if item.topic != "payments"]
        for item in found:
            topics.setdefault(item.topic, (item, part))
        if inquiry:
            unresolved.append(part)

    if any("환불" in part for part in unresolved):
        taste = next(item for item in RULES if item.topic == "taste_refund")
        if taste.question_pattern.search(text.replace("\n", " ")):
            topics.setdefault(taste.topic, (taste, text))
    matches = tuple(topics[item.topic] for item in RULES if item.topic in topics)
    covered = {part for _, part in matches}
    missing = tuple(part for part in unresolved if part not in covered and not (
        topics and FOLLOWUP_QUESTION.fullmatch(part)
    ))
    return matches, missing, ""


def questions_in(text: str) -> tuple[tuple[tuple[PolicyRule, str], ...], str]:
    """Return the rule-based router's decision for comparison and diagnostics."""
    matches, missing, reason = route_questions(text)
    if missing:
        return (), "unknown_question"
    return matches, reason
