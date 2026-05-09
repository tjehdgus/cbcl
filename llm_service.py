"""LLM 서비스 — OpenAI 호출, 입력·출력 가드레일, 메트릭 로깅."""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

from openai import OpenAI

from prompts import (
    SUMMARY_SYSTEM,
    SUMMARY_USER_TEMPLATE,
    QA_SYSTEM_TEMPLATE,
    GUARDRAIL_CHECK_PROMPT,
    HANDOFF_SYSTEM,
    HANDOFF_USER_TEMPLATE,
    CLINICAL_SIMPLIFY_SYSTEM,
    CLINICAL_SIMPLIFY_USER_TEMPLATE,
)
from quick_answers import lookup_quick_answer

logger = logging.getLogger("mameum.llm")


# ── 메트릭 로깅 ─────────────────────────────
_LOG_DIR = Path(__file__).resolve().parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_METRIC_LOG = _LOG_DIR / "metrics.jsonl"


def log_metric(event: str, **fields) -> None:
    """LLM 호출·가드레일·피드백 등 운영 이벤트를 JSON Lines로 기록.

    예:
        log_metric("easy_report", model="gpt-4o-mini",
                   prompt_tokens=2000, completion_tokens=1500,
                   elapsed_sec=18.3, violations=[])
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    try:
        with _METRIC_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("metric log failed: %s", e)


# ── 출력단 안전 필터: 진단·처방 표현 후처리 ──
# (regex pattern, replacement, severity)
_DX_TERMS = r"ADHD|adhd|에이디에이치디|우울증|불안장애|자폐(?:\s*스펙트럼)?|품행장애|적대적\s*반항(?:장애)?"
_DX_REPLACEMENT = "이 부분은 전문 상담사 선생님과 함께 자세히 확인해 보시면 좋겠어요"

SAFETY_PATTERNS = [
    # 진단명 + 단정 표현 (high)
    (rf"({_DX_TERMS})\s*(입니다|이에요|예요|입니다요|이네요|네요)", _DX_REPLACEMENT, "high"),
    # 진단명 + "(으)로 보입니다 / 의심됩니다 / 가능성" 형태 (high)
    (rf"({_DX_TERMS})(?:으|이)?로?\s*보(?:입니다|여요|여집니다|입니다요|여)", _DX_REPLACEMENT, "high"),
    (rf"({_DX_TERMS})(?:이|가)?\s*의심", _DX_REPLACEMENT, "high"),
    (rf"({_DX_TERMS})(?:일|할)?\s*가능성", _DX_REPLACEMENT, "high"),
    # 진단명 단독 노출 (medium) — paraphrase 회피 차단용
    (rf"\b({_DX_TERMS})\b", None, "medium"),
    # 처방·치료 권고 — 단정형 (high)
    (r"치료(?:가|를)?\s*(필요합?니다|반드시\s*필요|받으셔야)", "상담을 통해 더 자세히 살펴보시는 것을 권해드려요", "high"),
    (r"약(?:을|물을)?\s*(드셔야|복용해야|먹여야|복용을\s*권)\s*(합니다|해요|드립니다)?",
     "전문 상담을 통해 확인해 보시는 것이 좋겠어요", "high"),
    # 처방·치료 권고 — 회피형 (medium) — "병원 가보세요" 같이 단정 회피하는 표현
    (r"병원(?:에)?\s*(가셔야|가시는\s*것이\s*좋|가\s*보세요|가보세요|한\s*번\s*가)",
     "먼저 상담 선생님과 이야기해 보시면 좋겠어요", "medium"),
    (r"약(?:을|물을)?\s*(먹이는\s*게\s*좋|먹여\s*보세요|시도해\s*보세요)",
     "전문 상담을 통해 확인해 보시는 것이 좋겠어요", "medium"),
    # 단정적 위험·심각 판단 (medium)
    (r"심각합?니다", "조금 더 자세히 살펴볼 필요가 있어 보여요", "medium"),
    (r"위험합?니다", "주의 깊게 관찰이 필요할 수 있어요", "medium"),
]


def _normalize_for_safety(text: str) -> str:
    """안전 필터 매칭 전에 텍스트를 정규화.

    띄어쓰기 변형, 자모 분해 등을 잡기 위한 1차 정규화.
    실제 출력 텍스트는 변경하지 않고 매칭용 사본만 만든다.
    """
    # 자음/모음만 분리된 표기(예: "ㅇㅏㄷㅔㅇㅏㅇㅏㅎㅗㅏㄷㅡ") 정상화는 unicodedata.NFC로 충분
    import unicodedata
    return unicodedata.normalize("NFC", text)

SAFETY_NOTE = (
    "\n\n> ℹ️ **[안전 점검 안내]** 답변 일부 표현이 자동 점검되어 조정되었습니다. "
    "임상적 해석과 진단은 상담 선생님의 영역이며, 상담 시 함께 확인해 주세요."
)


def apply_safety_filter(text: str) -> Tuple[str, list]:
    """LLM 출력에서 진단·처방·단정 표현을 검출하여 부드러운 표현으로 치환.

    - replacement가 None인 패턴은 검출만 하고 텍스트는 보존 (medium 위반 기록 → SAFETY_NOTE 첨부)
    - 정규화(NFC) 후 매칭하여 자모 분해된 입력도 잡는다.

    Returns:
        (filtered_text, violations) — violations는 매칭된 항목 리스트
    """
    violations = []
    filtered = _normalize_for_safety(text)
    for idx, (pattern, replacement, severity) in enumerate(SAFETY_PATTERNS):
        matches = re.findall(pattern, filtered)
        if matches:
            violations.append({"idx": idx, "severity": severity, "count": len(matches)})
            if replacement is not None:
                filtered = re.sub(pattern, replacement, filtered)
    if violations:
        logger.warning("Safety filter triggered: %s", violations)
    return filtered, violations


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


def generate_easy_report(report_text: str, model: str = "gpt-4o-mini") -> str:
    """동기 버전 — SSE를 사용하지 않는 클라이언트(스크립트, 테스트)용."""
    t0 = time.time()
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(report_text=report_text)},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content
    filtered, violations = apply_safety_filter(raw)
    if any(v["severity"] == "high" for v in violations):
        filtered += SAFETY_NOTE

    usage = getattr(response, "usage", None)
    log_metric("easy_report", model=model,
               prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
               completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
               output_len=len(filtered),
               violations=[v for v in violations],
               elapsed_sec=round(time.time() - t0, 3))
    return filtered


def stream_easy_report(report_text: str, model: str = "gpt-4o-mini"):
    """SSE 스트리밍 버전 — 토큰 단위로 즉시 노출하여 체감 응답 속도 향상.

    스트림 종료 후 누적 출력에 안전 필터를 적용하여, high 위반이 있으면
    SAFETY_NOTE를 마지막에 추가로 yield 한다.
    """
    t0 = time.time()
    client = get_client()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": SUMMARY_USER_TEMPLATE.format(report_text=report_text)},
        ],
        temperature=0.3,
        max_tokens=2000,
        stream=True,
    )
    accumulated = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            accumulated += delta
            yield delta

    # 스트림 종료 후 안전 점검
    _, violations = apply_safety_filter(accumulated)
    if any(v["severity"] == "high" for v in violations):
        yield SAFETY_NOTE

    log_metric("easy_report_stream", model=model,
               output_len=len(accumulated),
               violations=[v for v in violations],
               elapsed_sec=round(time.time() - t0, 3))


def check_guardrail(question: str, model: str = "gpt-4o-mini") -> str:
    """입력 가드레일 — 의도 분류.

    Fail-safe 원칙: 분류 결과가 모호하거나 LLM 실패 시 'C'(상담사 안내)로 보낸다.
    'A'(통과)로 떨어지면 진단·처방 질문이 그대로 흘러갈 수 있어 위험하므로,
    민감 도메인(아동 정신건강)에서는 안전한 쪽으로 기본값을 잡는다.
    """
    # 1차: 명시적 진단명 키워드가 질문에 들어 있으면 LLM 호출 전에 차단
    diagnosis_pre_screen = re.compile(
        r"ADHD|adhd|에이디에이치디|"
        r"우울증|불안장애|자폐|품행장애|적대적\s*반항"
    )
    if diagnosis_pre_screen.search(question):
        return "C"

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": GUARDRAIL_CHECK_PROMPT.format(question=question)},
            ],
            temperature=0,
            max_tokens=5,
        )
        result = response.choices[0].message.content.strip().upper()
        for char in result:
            if char in "ABCD":
                return char
    except Exception as e:
        logger.warning("check_guardrail LLM call failed: %s", e)

    # Fail-safe: 분류 모호·실패 시 안전한 쪽으로 (상담사 안내)
    return "C"


def simplify_clinical_summary(interpretation_paragraphs: list,
                               observations: list,
                               model: str = "gpt-4o-mini") -> dict:
    """임상가 종합 해석 + 영역별 관찰 소견을 보호자 친화 한 단락으로 풀이.

    Returns:
        {"summary": "...", "key_points": ["...", "...", "..."]}
        실패 시 {"summary": "", "key_points": []}
    """
    if not interpretation_paragraphs and not observations:
        return {"summary": "", "key_points": []}

    t0 = time.time()
    interp_text = "\n\n".join(interpretation_paragraphs) if interpretation_paragraphs else "(없음)"
    obs_lines = []
    for o in observations:
        area = o.get("area", "")
        t_score = o.get("t_score", "")
        cls = o.get("classification", "")
        notes = o.get("notes", [])
        obs_lines.append(f"- {area} (T={t_score}, {cls})")
        for n in notes:
            obs_lines.append(f"  · {n}")
    obs_text = "\n".join(obs_lines) if obs_lines else "(없음)"

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CLINICAL_SIMPLIFY_SYSTEM},
            {"role": "user", "content": CLINICAL_SIMPLIFY_USER_TEMPLATE.format(
                interpretation=interp_text,
                observations=obs_text,
            )},
        ],
        temperature=0.3,
        max_tokens=900,
    )
    raw = response.choices[0].message.content.strip()

    # JSON 파싱
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("simplify_clinical JSON parse failed: %s", raw[:200])
        result = {"summary": "", "key_points": []}

    # 출력 안전 필터 적용
    summary = result.get("summary", "")
    if summary:
        filtered, violations = apply_safety_filter(summary)
        result["summary"] = filtered
        if any(v["severity"] == "high" for v in violations):
            result["summary"] += SAFETY_NOTE

    log_metric("simplify_clinical", model=model,
               output_len=len(result.get("summary", "")),
               key_points=len(result.get("key_points", [])),
               elapsed_sec=round(time.time() - t0, 3))
    return result


def generate_handoff_note(report_text: str, conversation_history: list, model: str = "gpt-4o-mini") -> str:
    """상담사 인계 메모 생성.

    보호자-챗봇 대화 누적을 분석해 상담사가 통화 전 빠르게 파악할 수 있는 메모를 생성한다.
    출력에도 동일한 안전 필터를 적용해 진단·처방 표현을 차단한다.
    """
    if not conversation_history:
        return "_대화 내역이 없습니다. 보호자와 Q&A를 먼저 진행한 후 다시 시도해 주세요._"

    t0 = time.time()
    convo_lines = []
    for m in conversation_history:
        role = "보호자" if m.get("role") == "user" else "AI 챗봇"
        convo_lines.append(f"[{role}] {m.get('content', '').strip()}")
    conversation_block = "\n".join(convo_lines)

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": HANDOFF_SYSTEM},
            {"role": "user", "content": HANDOFF_USER_TEMPLATE.format(
                report_context=report_text,
                conversation=conversation_block,
            )},
        ],
        temperature=0.2,
        max_tokens=1200,
    )
    raw = response.choices[0].message.content
    filtered, violations = apply_safety_filter(raw)
    if any(v["severity"] == "high" for v in violations):
        filtered += SAFETY_NOTE

    usage = getattr(response, "usage", None)
    log_metric("handoff_note", model=model,
               prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
               completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
               output_len=len(filtered),
               violations=[v for v in violations],
               elapsed_sec=round(time.time() - t0, 3))
    return filtered


def stream_chat_qa(report_text: str, conversation_history: list, user_question: str, model: str = "gpt-4o-mini"):
    """Q&A 스트리밍 제너레이터 (SSE용)

    처리 흐름:
      1. 빠른 답변 lookup — 정의형 질문이면 LLM 호출 없이 캐시 답변 반환
      2. 입력 가드레일 — C/D 카테고리는 차단
      3. LLM 스트리밍 — 공감·Citation·유사 질문 자동 포함
      4. 출력 가드레일 — 종료 후 안전 점검
    """
    t0 = time.time()

    # 1) 빠른 답변 캐시 (LLM skip)
    cached = lookup_quick_answer(user_question)
    if cached:
        log_metric("chat_quick_answer", question=user_question[:80],
                   elapsed_sec=round(time.time() - t0, 3))
        # 자연스러운 출력을 위해 chunk로 yield
        for line in cached.split("\n"):
            yield line + "\n"
        return

    # 2) 입력 가드레일
    category = check_guardrail(user_question, model)

    if category == "C":
        yield ("이 부분은 전문 상담사 선생님이 더 정확하게 안내해 주실 수 있는 영역이에요. 😊\n\n"
               "곧 있을 상담에서 이 궁금증을 꼭 말씀해 보세요. "
               "상담 선생님께서 아이의 상황에 맞는 구체적인 안내를 해 주실 거예요.")
        log_metric("chat_blocked_C", question=user_question[:80],
                   elapsed_sec=round(time.time() - t0, 3))
        return

    if category == "D":
        yield ("이 부분은 현재 보고서에 포함되지 않은 내용이에요.\n\n"
               "보고서에 나와 있는 내용에 대해서는 제가 도움을 드릴 수 있으니, "
               "다른 궁금한 점이 있으시면 편하게 물어봐 주세요! 😊")
        log_metric("chat_blocked_D", question=user_question[:80],
                   elapsed_sec=round(time.time() - t0, 3))
        return

    # 3) LLM 스트리밍
    client = get_client()
    system_prompt = QA_SYSTEM_TEMPLATE.format(report_text=report_text)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_question})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=900,  # citation + 유사 질문 추가로 약간 증가
        stream=True,
    )
    accumulated = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            accumulated += delta
            yield delta

    # 4) 출력 가드레일 — high 위반 시 보정 안내 추가
    _, violations = apply_safety_filter(accumulated)
    if any(v["severity"] == "high" for v in violations):
        yield SAFETY_NOTE

    log_metric("chat_qa", category=category, question=user_question[:80],
               output_len=len(accumulated),
               violations=[v for v in violations],
               elapsed_sec=round(time.time() - t0, 3))


# ── 유사 질문 추천은 QA_SYSTEM_TEMPLATE에 이미 통합되어 있음 ─────
# (보호자가 매번 같이 받게 됨 — 별도 LLM 호출 불필요)


def generate_followup_suggestions(report_text: str, conversation_history: list,
                                  model: str = "gpt-4o-mini") -> list[str]:
    """대화 종료 후 별도로 유사 질문 3개를 생성 (옵션).

    LLM 응답 안의 '💭 이런 것도 궁금하실 수 있어요'에서 자동 제안되긴 하지만,
    별도 API로 호출 시 사용 가능. 현 PoC는 응답 내장 방식 사용.
    """
    client = get_client()
    convo = "\n".join(f"[{m['role']}] {m['content']}"
                      for m in conversation_history[-6:])
    prompt = (
        f"보호자-AI Q&A 대화의 다음 단계로 추천할 후속 질문 3개를 제안하세요.\n"
        f"보고서 내용 범위 안에서, 보호자가 미처 묻지 않았지만 알면 도움될 주제로.\n"
        f"각 질문은 한 줄, JSON 배열로만 답하세요.\n\n"
        f"[보고서 요약]\n{report_text[:1000]}\n\n[최근 대화]\n{convo}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # JSON 파싱
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x).strip() for x in result[:3]]
    except Exception as e:
        logger.warning("followup gen failed: %s", e)
    return []
