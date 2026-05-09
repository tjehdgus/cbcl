"""PDF 파싱 자가 검증 — 7개 섹션 추출 정확도를 골든셋으로 자동 측정."""

import sys
import io
import json
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from pdf_parser import parse_cbcl_report

SAMPLE_PDF = Path(__file__).resolve().parents[2] / "AI개발자_테스트자료_CBCL보고서.pdf"


# ─────────────────────────────────────────────
# Ground Truth (샘플 보고서 PDF에서 직접 확인한 정답값)
# ─────────────────────────────────────────────
EXPECTED = {
    "child_name": "김인사",
    "gender": "남아",
    "age_substr": "7세",  # "만 7세 3개월"의 일부
    "test_date": "2026.03.12",
    "norm_substr": "남자 4",  # "남자 4–11세"의 일부
    "internalizing_t": 61,
    "externalizing_t": 54,
    "total_t": 58,
    "scales_count": 8,
    "scale_t_scores": {
        "위축": 60,
        "신체증상": 55,
        "우울/불안": 63,
        "사회적 미성숙": 62,
        "사고의 문제": 57,
        "주의집중 문제": 66,
        "비행": 52,
        "공격성": 50,
    },
    "scale_groups": {
        "internalizing": {"위축", "신체증상", "우울/불안"},
        "mixed": {"사회적 미성숙", "사고의 문제", "주의집중 문제"},
        "externalizing": {"비행", "공격성"},
    },
    "parent_comments_count": 2,
    "special_scales_count": 2,
    "special_scale_names": {"정서불안정 척도", "성문제 척도"},
    "observations_count": 5,
    "observations_areas": {
        "내재화 문제 종합", "주의집중 문제", "우울/불안",
        "사회적 미성숙", "위축",
    },
    "comprehensive_interpretation_min_chars": 200,
}


def run():
    if not SAMPLE_PDF.exists():
        print(f"[FAIL] 샘플 PDF 없음: {SAMPLE_PDF}")
        raise SystemExit(1)

    report = parse_cbcl_report(str(SAMPLE_PDF))

    checks = []

    def check(name, actual, ok, note=""):
        checks.append({"name": name, "actual": str(actual)[:80], "passed": bool(ok), "note": note})

    # 1. 아동 정보
    check("아동 이름", report.child_name, report.child_name == EXPECTED["child_name"])
    check("성별", report.gender, report.gender == EXPECTED["gender"])
    check("연령 (substr)", report.age, EXPECTED["age_substr"] in report.age)
    check("검사일", report.test_date, report.test_date == EXPECTED["test_date"])
    check("규준 (substr)", report.norm_group, EXPECTED["norm_substr"] in report.norm_group)

    # 2. 종합 지표
    check("내재화 T", report.internalizing_t, report.internalizing_t == EXPECTED["internalizing_t"])
    check("외현화 T", report.externalizing_t, report.externalizing_t == EXPECTED["externalizing_t"])
    check("총 문제행동 T", report.total_t, report.total_t == EXPECTED["total_t"])

    # 3. 8개 척도
    check("척도 개수", len(report.scales), len(report.scales) == EXPECTED["scales_count"])
    actual_t = {s.name: s.t_score for s in report.scales}
    for name, expected_t in EXPECTED["scale_t_scores"].items():
        check(f"척도 '{name}' T점수", actual_t.get(name), actual_t.get(name) == expected_t)

    # 척도 그룹 분포
    actual_groups = {}
    for s in report.scales:
        actual_groups.setdefault(s.group, set()).add(s.name)
    for group, expected_names in EXPECTED["scale_groups"].items():
        check(f"그룹 '{group}'", actual_groups.get(group), actual_groups.get(group) == expected_names)

    # 4. 보호자 의견
    check("보호자 의견 개수", len(report.parent_comments),
          len(report.parent_comments) == EXPECTED["parent_comments_count"])

    # 5. 특수 척도 (Ⅳ)
    check("특수 척도 개수", len(report.special_scales),
          len(report.special_scales) == EXPECTED["special_scales_count"])
    actual_sp_names = {s.name for s in report.special_scales}
    check("특수 척도 이름", actual_sp_names, actual_sp_names == EXPECTED["special_scale_names"])

    # 6. 주요 관찰 소견 (Ⅴ)
    check("관찰 소견 개수", len(report.observations),
          len(report.observations) == EXPECTED["observations_count"])
    actual_obs_areas = {o.area for o in report.observations}
    check("관찰 소견 영역", actual_obs_areas, actual_obs_areas == EXPECTED["observations_areas"])
    # notes도 잘 추출됐는지
    obs_with_notes = sum(1 for o in report.observations if o.notes)
    check("관찰 소견 notes 추출률",
          f"{obs_with_notes}/{len(report.observations)}",
          obs_with_notes >= 4,  # 최소 4개 이상 notes 가지고 있어야 함
          note="≥4 expected")

    # 7. 종합 해석 (Ⅵ)
    interp_chars = sum(len(p) for p in report.comprehensive_interpretation)
    check("종합 해석 추출",
          f"{len(report.comprehensive_interpretation)} 단락, {interp_chars}자",
          interp_chars >= EXPECTED["comprehensive_interpretation_min_chars"],
          note=f"≥{EXPECTED['comprehensive_interpretation_min_chars']}자 expected")

    # 결과 집계
    total = len(checks)
    passed = sum(1 for c in checks if c["passed"])
    rate = passed / total

    print("=" * 70)
    print("PDF 파싱 자가 검증 — 샘플 K-CBCL 보고서")
    print("=" * 70)
    for c in checks:
        mark = "PASS" if c["passed"] else "FAIL"
        line = f"  [{mark}] {c['name']:25s} | actual: {c['actual']}"
        if c["note"]:
            line += f"   ({c['note']})"
        print(line)

    print(f"\n[종합]")
    print(f"  검사 항목: {total}개")
    print(f"  통과: {passed}개 ({rate:.1%})")
    print(f"  PDF 7개 섹션 추출 정확도: {'합격' if rate >= 0.95 else '미달'} (합격선 95%)")

    out = {
        "summary": {"total": total, "passed": passed, "rate": round(rate, 4)},
        "checks": checks,
    }
    out_path = Path(__file__).parent / "eval_pdf_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[저장] {out_path.name}")


if __name__ == "__main__":
    run()
