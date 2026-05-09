"""PDF 파서 — pdfplumber + 정규식으로 K-CBCL 7개 섹션 추출."""

import re
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber


@dataclass
class ScaleScore:
    """개별 척도 점수"""
    name: str
    name_en: str
    t_score: int
    category: str
    signal: str
    group: str = ""  # "internalizing" | "mixed" | "externalizing"


@dataclass
class SpecialScale:
    """특수 척도 (한국판 고유) — Ⅳ. 특수 척도 섹션."""
    name: str           # "정서불안정 척도" / "성문제 척도"
    age_range: str      # "6-11세" / "4-11세"
    status: str         # "미실시" / "실시"
    note: str           # 추후 실시 권장 등


@dataclass
class Observation:
    """주요 관찰 소견 — Ⅴ. 주요 관찰 소견 섹션."""
    area: str               # "내재화 문제 종합" / "주의집중 문제" 등
    t_score: int            # 61 / 66 / ...
    classification: str     # "종합지표 준임상" / "개별척도 준임상 (60-69T)"
    notes: list = field(default_factory=list)  # 영역별 관찰 항목들


@dataclass
class CBCLReport:
    """CBCL 보고서 구조화 데이터."""
    child_name: str = ""
    gender: str = ""
    age: str = ""
    test_date: str = ""
    norm_group: str = ""
    internalizing_t: Optional[int] = None
    internalizing_category: str = ""
    externalizing_t: Optional[int] = None
    externalizing_category: str = ""
    total_t: Optional[int] = None
    total_category: str = ""
    scales: list = field(default_factory=list)
    parent_comments: list = field(default_factory=list)
    # ── 신규 (v6) ──
    special_scales: list = field(default_factory=list)        # Ⅳ. 특수 척도
    observations: list = field(default_factory=list)          # Ⅴ. 주요 관찰 소견
    comprehensive_interpretation: list = field(default_factory=list)  # Ⅵ. 종합 해석 (단락 리스트)
    raw_text: str = ""


def classify_score(t_score: int, is_composite: bool = False) -> tuple:
    if is_composite:
        if t_score < 60:
            return "정상", "🟢"
        elif t_score <= 62:
            return "준임상", "🟡"
        else:
            return "임상", "🟠"
    else:
        if t_score < 60:
            return "정상", "🟢"
        elif t_score <= 69:
            return "준임상", "🟡"
        else:
            return "임상", "🟠"


def extract_text_from_pdf(pdf_path: str) -> str:
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n\n".join(full_text)


def parse_cbcl_report(pdf_path: str) -> CBCLReport:
    raw_text = extract_text_from_pdf(pdf_path)
    report = CBCLReport(raw_text=raw_text)

    # 기본 정보
    info_match = re.search(
        r"이름\s+성별\s*/\s*연령\s+적용\s*규준\s+검사일\s*\n"
        r"([가-힣]+)\s+(남아|여아)\s*/\s*만\s*(\d+세\s*\d*개월?)\s+"
        r"([가-힣0-9\s–\-]+?)\s+(\d{4}\.\d{2}\.\d{2})",
        raw_text,
    )
    if info_match:
        report.child_name = info_match.group(1)
        report.gender = info_match.group(2)
        report.age = "만 " + info_match.group(3)
        report.norm_group = info_match.group(4).strip()
        report.test_date = info_match.group(5)
    else:
        nm = re.search(r"\n([가-힣]{2,4})\s+(남아|여아)", raw_text)
        if nm:
            report.child_name = nm.group(1)
            report.gender = nm.group(2)
        ga = re.search(r"(남아|여아)\s*/\s*만\s*(\d+세\s*\d*개월?)", raw_text)
        if ga:
            report.gender = ga.group(1)
            report.age = "만 " + ga.group(2)
        dm = re.search(r"(\d{4}\.\d{2}\.\d{2})", raw_text)
        if dm:
            report.test_date = dm.group(1)

    # 종합 지표
    cb = re.search(
        r"내재화\s*문제\s+외현화\s*문제\s+총\s*문제행동.*?\n.*?\n(\d{2,3})\s+(\d{2,3})\s+(\d{2,3})",
        raw_text, re.DOTALL,
    )
    if cb:
        report.internalizing_t = int(cb.group(1))
        report.externalizing_t = int(cb.group(2))
        report.total_t = int(cb.group(3))
    else:
        for label, attr in [("내재화", "internalizing_t"), ("외현화", "externalizing_t"), ("총 문제행동", "total_t")]:
            m = re.search(rf"{label}[^(]*\(T\s*=\s*(\d+)", raw_text)
            if m:
                setattr(report, attr, int(m.group(1)))

    for attr in ["internalizing_t", "externalizing_t", "total_t"]:
        t = getattr(report, attr)
        if t is not None:
            cat, _ = classify_score(t, is_composite=True)
            setattr(report, attr.replace("_t", "_category"), cat)

    # 개별 척도 (page 2 layout) — 그룹 정보 포함
    scale_defs = [
        (r"위축\n(\d{2,3})\s+(정상|준임상)", "위축", "Withdrawn", "internalizing"),
        (r"신체증상\n(\d{2,3})\s+(정상|준임상)", "신체증상", "Somatic Complaints", "internalizing"),
        (r"우울/불안\n(\d{2,3})\s+(정상|준임상)", "우울/불안", "Depressed/Anxious", "internalizing"),
        (r"사회적 미성숙\n(\d{2,3})\s+(정상|준임상)", "사회적 미성숙", "Social Immaturity", "mixed"),
        (r"사고의 문제\n(\d{2,3})\s+(정상|준임상)", "사고의 문제", "Thought Problems", "mixed"),
        (r"주의집중 문제\n(\d{2,3})\s+(정상|준임상)", "주의집중 문제", "Attention Problems", "mixed"),
        (r"비행\n(\d{2,3})\s+(정상|준임상)", "비행", "Delinquent", "externalizing"),
        (r"공격성\n(\d{2,3})\s+(정상|준임상)", "공격성", "Aggressive", "externalizing"),
    ]

    for pattern, name_kr, name_en, group in scale_defs:
        match = re.search(pattern, raw_text)
        if match:
            t_score = int(match.group(1))
            category, signal = classify_score(t_score, is_composite=False)
            report.scales.append(ScaleScore(name_kr, name_en, t_score, category, signal, group))

    # 폴백
    if not report.scales:
        fb = [
            (r"위축.*?T\s*=\s*(\d+)", "위축", "Withdrawn"),
            (r"우울/?불안.*?T\s*=\s*(\d+)", "우울/불안", "Depressed/Anxious"),
            (r"사회적\s*미성숙.*?T\s*=\s*(\d+)", "사회적 미성숙", "Social Immaturity"),
            (r"주의집중\s*문제.*?T\s*=\s*(\d+)", "주의집중 문제", "Attention Problems"),
        ]
        for pattern, name_kr, name_en in fb:
            match = re.search(pattern, raw_text)
            if match:
                t_score = int(match.group(1))
                category, signal = classify_score(t_score, is_composite=False)
                report.scales.append(ScaleScore(name_kr, name_en, t_score, category, signal))

    # 보호자 의견
    ps = re.search(r"보호자\s*참고\s*의견(.*?)해석\s*시\s*유의사항", raw_text, re.DOTALL)
    if ps:
        comments = re.findall(r'["\u201C\u201D]([^"\u201C\u201D]+)["\u201C\u201D]', ps.group(1))
        report.parent_comments = comments

    _extract_extra_sections(raw_text, report)
    return report


def _extract_extra_sections(raw_text: str, report: "CBCLReport") -> None:
    """\u2163 \uD2B9\uC218 \uCC99\uB3C4 / \u2164 \uC8FC\uC694 \uAD00\uCC30 \uC18C\uACAC / \u2165 \uC885\uD569 \uD574\uC11D \uCD94\uCD9C (in-place)."""
    # \u2500\u2500 \u2163. \uD2B9\uC218 \uCC99\uB3C4 \u2500\u2500
    sp_section = re.search(r"[\u2163IV4]\.\s*\uD2B9\uC218\s*\uCC99\uB3C4.*?(?=[\u2164V5]\.|$)", raw_text, re.DOTALL)
    if sp_section:
        # \uB2E4\uC74C \uCC99\uB3C4 \uB610\uB294 \uC139\uC158 \uB05D\uAE4C\uC9C0\uB97C note\uB85C \uBC1B\uC74C (multi-line, \uC904\uBC14\uAFC8\uC740 \uACF5\uBC31\uC73C\uB85C)
        for m in re.finditer(
            r"(\uC815\uC11C\uBD88\uC548\uC815\s*\uCC99\uB3C4|\uC131\uBB38\uC81C\s*\uCC99\uB3C4)\s*\(([^)]+)\)\s*[:\uFF1A]\s*"
            r"(\uBBF8\uC2E4\uC2DC|\uC2E4\uC2DC)\.\s*"
            r"(.*?)(?=\n*\s*(?:\uC815\uC11C\uBD88\uC548\uC815\s*\uCC99\uB3C4|\uC131\uBB38\uC81C\s*\uCC99\uB3C4|$))",
            sp_section.group(0),
            re.DOTALL,
        ):
            paren = m.group(2).strip()
            age_m = re.search(r"(\d+\s*[\-\u2013~]\s*\d+\s*\uC138)", paren)
            # \uC904\uBC14\uAFC8\uC73C\uB85C \uB2E8\uC5B4\uAC00 \uC798\uB9B0 \uACBD\uC6B0 \uACF5\uBC31 \uC5C6\uC774 \uD569\uCE58\uAE30 ("\uD574\n\uB2F9 \uC5F0\uB839\uAD70" \u2192 "\uD574\uB2F9 \uC5F0\uB839\uAD70")
            # PDF\uC5D0\uC11C \uD55C \uC904 \uB05D\uC5D0\uC11C \uB2E4\uC74C \uC904\uB85C \uB118\uC5B4\uAC08 \uB54C \uBCF4\uD1B5 \uB2E8\uC5B4\uAC00 \uC798\uB9AC\uBBC0\uB85C \uACF5\uBC31 X\uB85C \uD569\uCE68
            raw_note = m.group(4).strip()
            note = re.sub(r"\s*\n\s*", "", raw_note)
            note = re.sub(r"[ \t]+", " ", note).strip()
            report.special_scales.append(SpecialScale(
                name=m.group(1).strip(),
                age_range=age_m.group(1) if age_m else paren,
                status=m.group(3).strip(),
                note=note,
            ))

    # \u2500\u2500 \u2164. \uC8FC\uC694 \uAD00\uCC30 \uC18C\uACAC \u2500\u2500
    obs_section = re.search(r"[\u2164V5]\.\s*\uC8FC\uC694\s*\uAD00\uCC30\s*\uC18C\uACAC(.*?)(?=[\u2165VI6]\.|$)", raw_text, re.DOTALL)
    if obs_section:
        current = None
        for raw_line in obs_section.group(1).split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            header = re.match(r"([\uAC00-\uD7A3/\s]+?)\s*[\u00B7\u2022\u30FB]\s*T\s*=\s*(\d+)\s+(.+)", line)
            if header:
                if current:
                    report.observations.append(current)
                current = Observation(
                    area=header.group(1).strip(),
                    t_score=int(header.group(2)),
                    classification=header.group(3).strip(),
                    notes=[],
                )
            elif current is not None:
                current.notes.append(line)
        if current:
            report.observations.append(current)

    # \u2500\u2500 \u2165. \uC885\uD569 \uD574\uC11D \u2500\u2500
    interp_section = re.search(r"[\u2165VI6]\.\s*\uC885\uD569\s*\uD574\uC11D(.*?)(?=[\u2166VII7]\.|$)", raw_text, re.DOTALL)
    if interp_section:
        text_blk = interp_section.group(1).strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text_blk) if p.strip()]
        if len(paragraphs) <= 1 and text_blk:
            paragraphs = [text_blk]
        report.comprehensive_interpretation = paragraphs


# 보고서 원문 용어 → 보호자가 이해할 수 있는 일상어 풀이 매핑.
# LLM 컨텍스트에 함께 주입해, 출력에서 원문 용어가 단독으로 등장하지 않도록 한다.
COMPOSITE_FRIENDLY = {
    "내재화": "감정·기분의 영역 (걱정·위축·신체 호소 등 안으로 드러나는 어려움)",
    "외현화": "행동·표현의 영역 (다툼·규칙 위반·공격성 등 밖으로 드러나는 어려움)",
    "총": "전체 합계 (감정·행동 두 영역을 함께 종합)",
}

SCALE_FRIENDLY = {
    "위축": "혼자 있기를 좋아하거나 무기력함을 보이는 정도",
    "신체증상": "두통·복통처럼 몸의 증상으로 호소하는 정도",
    "우울/불안": "슬픔·걱정·두려움 같은 감정의 정도",
    "사회적 미성숙": "또래 관계 형성과 상호작용의 어려움 정도",
    "사고의 문제": "강박적이거나 비현실적인 생각의 정도",
    "주의집중 문제": "주의를 지속하거나 산만함을 조절하는 어려움 정도",
    "비행": "규칙 위반·거짓말 등 사회 규범에서 벗어나는 행동 정도",
    "공격성": "다툼·화내기·파괴적 행동의 정도",
}


def report_to_context_string(report: CBCLReport) -> str:
    lines = [
        "## 아동 정보",
        f"- 이름: {report.child_name}",
        f"- 성별/연령: {report.gender} / {report.age}",
        f"- 검사일: {report.test_date}",
        f"- 적용 규준: {report.norm_group}",
        "",
        "## 종합 지표 (보호자 친화 라벨 → 보고서 원문 용어)",
        f"- 감정·기분의 영역 (내재화): T={report.internalizing_t} → {report.internalizing_category}",
        f"- 행동·표현의 영역 (외현화): T={report.externalizing_t} → {report.externalizing_category}",
        f"- 전체 합계 (총 문제행동): T={report.total_t} → {report.total_category}",
        "",
        "## 개별 척도 (이름 + 일상어 풀이)",
    ]
    for s in report.scales:
        friendly = SCALE_FRIENDLY.get(s.name, "")
        if friendly:
            lines.append(f'- "{s.name}" — {friendly}: T={s.t_score} → {s.category} {s.signal}')
        else:
            lines.append(f"- {s.name} ({s.name_en}): T={s.t_score} → {s.category} {s.signal}")
    if report.special_scales:
        lines.append("")
        lines.append("## 특수 척도 (한국판 고유)")
        for s in report.special_scales:
            lines.append(f"- {s.name} ({s.age_range}): {s.status} — {s.note}")

    if report.observations:
        lines.append("")
        lines.append("## 주요 관찰 소견 (임상가 평어)")
        for o in report.observations:
            lines.append(f"### {o.area} (T={o.t_score}, {o.classification})")
            for note in o.notes:
                lines.append(f"- {note}")

    if report.comprehensive_interpretation:
        lines.append("")
        lines.append("## 종합 해석 (임상가 종합 평가)")
        for para in report.comprehensive_interpretation:
            lines.append(para)
            lines.append("")

    if report.parent_comments:
        lines.append("")
        lines.append("## 보호자 참고 의견")
        for c in report.parent_comments:
            lines.append(f'- "{c}"')
    lines.append("")
    lines.append("## K-CBCL 해석 기준 참고")
    lines.append("- 종합척도: T<60 정상, T60-62 준임상, T>=63 임상")
    lines.append("- 개별 증후군: T<60 정상, T60-69 준임상, T>=70 임상")
    lines.append("- T점수: 평균 50, 표준편차 10. 높을수록 해당 영역의 문제행동이 또래보다 많이 보고됨을 의미")
    lines.append("")
    lines.append("## 보호자 표기 정책 (LLM이 보호자에게 전달할 때 반드시 지킬 규칙)")
    lines.append("- '내재화 문제'를 단독으로 쓰지 말 것 → '감정·기분의 영역(내재화)' 형식 (괄호 안은 짧게 한 단어)")
    lines.append("- '외현화 문제' → '행동·표현의 영역(외현화)' 형식")
    lines.append("- '총 문제행동' → '전체 합계(총 문제행동)' 형식")
    lines.append("- '준임상' → '또래보다 조금 높은 편'")
    lines.append("- '임상' → '상담에서 함께 확인이 권장되는 수준'")
    lines.append("- 척도명도 처음 등장 시 짧은 풀이 괄호 함께 표기. 예: '위축(혼자 있기 선호)', '우울/불안(슬픔·걱정)'")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        report = parse_cbcl_report(sys.argv[1])
        print(report_to_context_string(report))
    else:
        print("Usage: python pdf_parser.py <path_to_cbcl_pdf>")
