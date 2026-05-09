"""FastAPI 백엔드 — 6개 엔드포인트, .env 자동 로딩, SSE 스트리밍."""

import os
import sys
import json
import tempfile
from contextlib import asynccontextmanager

# Windows 한국어 환경(cp949)에서 stdout 이모지·한글 출력 시 UnicodeEncodeError 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

load_dotenv()

from pdf_parser import parse_cbcl_report, report_to_context_string
from llm_service import (
    generate_easy_report,
    stream_easy_report,
    stream_chat_qa,
    generate_handoff_note,
    simplify_clinical_summary,
)
from care_signals import detect_care_signal


# ── Pydantic 모델 ──
class EasyReportRequest(BaseModel):
    report_context: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    report_context: str
    conversation_history: list[ChatMessage] = []
    question: str


class CareCheckRequest(BaseModel):
    conversation_history: list[ChatMessage] = []


class HandoffRequest(BaseModel):
    report_context: str
    conversation_history: list[ChatMessage] = []


class FeedbackRequest(BaseModel):
    """답변 후 보호자가 남기는 피드백 (좋아요/별로 + 짧은 코멘트)"""
    rating: str  # "up" | "down"
    question: str = ""
    answer: str = ""
    comment: str = ""


class SimplifyClinicalRequest(BaseModel):
    """임상가 종합 해석·관찰 소견을 보호자 친화 풀이로 변환 요청."""
    interpretation: list[str] = []
    observations: list[dict] = []


# ── 앱 설정 ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🤝 맘이음 API 서버 시작")
    print(f"   API docs: http://localhost:8000/docs")
    print(f"   프론트엔드: http://localhost:8000")
    yield
    print("맘이음 API 서버 종료")


app = FastAPI(
    title="맘이음 API",
    description="CBCL 보고서 AI 해설 서비스",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API 엔드포인트 ──

@app.post("/api/upload")
async def upload_report(file: UploadFile = File(...)):
    """CBCL 보고서 PDF 업로드 및 파싱"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report = parse_cbcl_report(tmp_path)
        context = report_to_context_string(report)

        return {
            "success": True,
            "child_info": {
                "name": report.child_name,
                "gender": report.gender,
                "age": report.age,
                "test_date": report.test_date,
                "norm_group": report.norm_group,
            },
            "composite_scores": {
                "internalizing": {"t_score": report.internalizing_t, "category": report.internalizing_category},
                "externalizing": {"t_score": report.externalizing_t, "category": report.externalizing_category},
                "total": {"t_score": report.total_t, "category": report.total_category},
            },
            "scales": [
                {"name": s.name, "name_en": s.name_en, "t_score": s.t_score,
                 "category": s.category, "signal": s.signal, "group": s.group}
                for s in report.scales
            ],
            "parent_comments": report.parent_comments,
            "special_scales": [
                {"name": s.name, "age_range": s.age_range, "status": s.status, "note": s.note}
                for s in report.special_scales
            ],
            "observations": [
                {"area": o.area, "t_score": o.t_score,
                 "classification": o.classification, "notes": o.notes}
                for o in report.observations
            ],
            "comprehensive_interpretation": report.comprehensive_interpretation,
            "report_context": context,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"보고서 파싱 실패: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/api/easy-report")
async def create_easy_report(req: EasyReportRequest):
    """보고서 풀어보기 — SSE 스트리밍으로 토큰 단위 응답.

    체감 응답 속도 향상: 첫 토큰까지 ~1-2초, 전체는 백그라운드에서 계속 흘러옴.
    """
    def event_stream():
        try:
            for chunk in stream_easy_report(req.report_context):
                data = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Q&A 챗봇 (SSE 스트리밍)"""
    history = [{"role": m.role, "content": m.content} for m in req.conversation_history]

    def event_stream():
        for chunk in stream_chat_qa(req.report_context, history, req.question):
            data = json.dumps({"content": chunk}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/care-signal-check")
async def care_signal_check(req: CareCheckRequest):
    """보호자 누적 질문에서 위기 시그널을 검출하고 케어 제안을 반환."""
    history = [{"role": m.role, "content": m.content} for m in req.conversation_history]
    return detect_care_signal(history)


@app.post("/api/counselor-handoff")
async def counselor_handoff(req: HandoffRequest):
    """보호자-챗봇 대화를 바탕으로 상담사용 인계 메모를 생성."""
    history = [{"role": m.role, "content": m.content} for m in req.conversation_history]
    try:
        note = generate_handoff_note(req.report_context, history)
        return {"success": True, "content": note}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simplify-clinical")
async def simplify_clinical(req: SimplifyClinicalRequest):
    """임상가 종합 해석·영역별 관찰 소견을 보호자 친화 한 단락 + 핵심 포인트 3개로 풀이."""
    try:
        result = simplify_clinical_summary(req.interpretation, req.observations)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """답변 만족도 피드백 (👍/👎)을 구조화 로그로 수집.

    수집 목적:
      - 보호자가 어떤 답변에 만족·불만족했는지 트래킹
      - sLLM 파인튜닝 시 high-quality 응답 페어 시드로 활용
      - 가드레일 false positive 발견 (정상 응답인데 👎 받은 경우)
    """
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    from llm_service import log_metric
    log_metric("feedback",
               rating=req.rating,
               question=req.question[:200],
               answer_len=len(req.answer),
               comment=req.comment[:200])
    return {"success": True}


@app.get("/api/health")
async def health():
    """헬스 체크"""
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {"status": "ok", "openai_key_configured": has_key}


# ── 프론트엔드 정적 파일 서빙 ──
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/assets", StaticFiles(directory=frontend_dir), name="frontend-assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dir, "index.html"))
