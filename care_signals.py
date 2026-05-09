"""보호자 위기 시그널 감지 — 누적 Q&A에서 Regex + LLM 2단으로 검출."""

import logging
import os
import re

from openai import OpenAI

logger = logging.getLogger("mameum.care")


# ── 1단: 키워드 패턴 ──
# 즉각 위기 신호 (자해·극단적 표현)
HARD_CRISIS_PATTERNS = [
    r"자해", r"죽고\s*싶", r"극단(적|적인\s*선택)",
    r"포기하고\s*싶", r"버틸\s*수\s*없", r"숨\s*막혀\s*죽",
]

# 보호자 정서 어려움 신호 (중간)
SOFT_CRISIS_PATTERNS = [
    r"잠도?\s*못\s*자", r"너무\s*불안", r"우울해",
    r"무서워(요|서)", r"두려워(요|서)", r"혼자\s*감당",
    r"눈물.*나", r"답답해(요|서)", r"막막(해|함)",
    r"감당.*안\s*돼", r"심장.*두근",
]

# 강조어 (개수가 많을수록 정서 강도 ↑)
INTENSIFIERS = [r"진짜", r"정말", r"너무", r"심각", r"확실히", r"제발"]


def _scan_keywords(text: str) -> dict:
    """단일 텍스트에서 키워드 매칭 결과를 반환."""
    return {
        "hard": [p for p in HARD_CRISIS_PATTERNS if re.search(p, text)],
        "soft": [p for p in SOFT_CRISIS_PATTERNS if re.search(p, text)],
        "intensifiers": sum(1 for p in INTENSIFIERS if re.search(p, text)),
    }


def _keyword_level(conversation: list[dict]) -> tuple[str, list[str]]:
    """누적 보호자 메시지에서 키워드 기반 위기 수준을 산정.

    Returns:
        (level, matched_keywords)
        level: "danger" | "warning" | "low" | "normal"
    """
    user_texts = [m["content"] for m in conversation if m.get("role") == "user"]
    if not user_texts:
        return "normal", []

    # 최근 5개 메시지 누적 분석
    recent = " ".join(user_texts[-5:])
    scan = _scan_keywords(recent)

    if scan["hard"]:
        return "danger", scan["hard"]

    soft_count = len(scan["soft"])
    if soft_count >= 2 or (soft_count >= 1 and scan["intensifiers"] >= 2):
        return "warning", scan["soft"]
    if soft_count >= 1:
        return "low", scan["soft"]
    return "normal", []


# ── 2단: LLM 분류 ──
_CARE_CHECK_PROMPT = """다음은 K-CBCL 검사 보고서를 받은 아동 보호자가 AI 챗봇에 한 질문 목록입니다.
보호자(아동이 아닌 보호자 본인)의 정서 상태를 평가하세요.

질문 목록:
{questions}

평가 기준:
- DANGER: 보호자가 자해·극단 표현을 직접 했거나, 본인이 정서적으로 무너지는 신호
- WARNING: 보호자가 명확한 불안·우울·두려움·무력감을 반복 표현
- LOW: 일반 수준의 걱정·궁금증
- NORMAL: 정보 요청·일반 호기심

DANGER, WARNING, LOW, NORMAL 중 하나로만 답하세요."""


def _llm_level(conversation: list[dict], model: str = "gpt-4o-mini") -> str:
    """LLM 기반 미세 분류 (DANGER/WARNING/LOW/NORMAL)."""
    user_texts = [m["content"] for m in conversation if m.get("role") == "user"]
    if not user_texts:
        return "NORMAL"

    questions_block = "\n".join(f"- {q}" for q in user_texts[-5:])

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "NORMAL"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": _CARE_CHECK_PROMPT.format(questions=questions_block)}],
            temperature=0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().upper()
        for level in ("DANGER", "WARNING", "LOW", "NORMAL"):
            if level in result:
                return level
    except Exception as e:
        logger.warning("LLM care check failed: %s", e)
    return "NORMAL"


# ── 통합 검출 ──
_LEVEL_ORDER = ["normal", "low", "warning", "danger"]


def detect_care_signal(conversation: list[dict]) -> dict:
    """보호자 대화에서 위기 시그널을 검출.

    Args:
        conversation: [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        {
            "level": "normal" | "low" | "warning" | "danger",
            "keywords": [...],            # 매칭된 키워드 패턴
            "suggestion": str | None,     # 보호자에게 노출할 제안 문구
            "action": str | None,         # 권장 시스템 액션 ID
        }
    """
    if not conversation:
        return {"level": "normal", "keywords": [], "suggestion": None, "action": None}

    # 1단: 키워드 매칭 (비용 0)
    kw_level, kw_matches = _keyword_level(conversation)

    # 2단: LLM 분류 — 키워드에서 신호 있거나 누적 질문 4회 이상일 때만 호출
    user_count = sum(1 for m in conversation if m.get("role") == "user")
    llm_level = "NORMAL"
    if kw_level in ("warning", "danger") or user_count >= 4:
        llm_level = _llm_level(conversation)

    # 두 결과 중 더 높은 쪽 채택
    final_level = max(
        kw_level, llm_level.lower(),
        key=lambda x: _LEVEL_ORDER.index(x) if x in _LEVEL_ORDER else 0,
    )

    if final_level == "danger":
        return {
            "level": "danger",
            "keywords": kw_matches,
            "suggestion": (
                "보호자님께서도 지금 많이 힘드신 것 같아요. 잠시 멈추시고, 가능하시다면 "
                "예정된 상담을 앞당기시거나 24시간 정신건강 상담전화(1577-0199)로 "
                "지금 도움을 받아보시는 것을 권해드려요."
            ),
            "action": "expedite_or_emergency",
        }
    if final_level == "warning":
        return {
            "level": "warning",
            "keywords": kw_matches,
            "suggestion": (
                "보호자님께서 많이 걱정이 되시는 것 같아요. 곧 있을 상담에서 아이 일뿐 아니라 "
                "보호자님의 걱정도 함께 이야기해 보시면 도움이 될 거예요. "
                "필요하시면 상담 일정을 앞당겨 드릴 수도 있어요."
            ),
            "action": "offer_reschedule",
        }
    if final_level == "low":
        return {"level": "low", "keywords": kw_matches, "suggestion": None, "action": None}
    return {"level": "normal", "keywords": [], "suggestion": None, "action": None}


if __name__ == "__main__":
    # 빠른 자가 테스트 (LLM 호출 없이 키워드만)
    test_cases = [
        [{"role": "user", "content": "T점수가 뭔가요?"}],
        [{"role": "user", "content": "주의집중이 너무 걱정돼요. 잠도 못 자요."}],
        [{"role": "user", "content": "솔직히 너무 불안하고 답답해요. 정말 무서워요."}],
        [{"role": "user", "content": "도저히 감당이 안 돼요. 죽고 싶을 정도로 힘들어요."}],
    ]
    for case in test_cases:
        result = detect_care_signal(case)
        print(f"Q: {case[0]['content']}")
        print(f"   level={result['level']}, kw={result['keywords']}")
        print(f"   action={result['action']}")
        print()
