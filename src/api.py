import sys
from pathlib import Path

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from generator import ask_rag_stream, ask_rag_with_trace
from guardrail import (
    DEFAULT_MIN_BM25_SCORE,
    DEFAULT_MIN_DENSE_SCORE,
    DEFAULT_MIN_RERANK_SCORE,
)
from indexer import (
    MAX_UPLOAD_SIZE_BYTES,
    delete_document as indexer_delete_document,
    list_documents as indexer_list_documents,
    save_and_index_document,
    validate_filename,
)
from llm_client import MODEL_NAME
from reranker import RERANK_MODEL
from retriever import EMBEDDING_MODEL, get_qdrant_health, search
from session_store import DEFAULT_SESSION_DB_PATH, SqliteSessionStore

logger = logging.getLogger(__name__)

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

SessionStore = SqliteSessionStore


class ChunkResult(BaseModel):
    chunk_id: str
    source: str
    section: str
    text: str
    score: float
    dense_rank: int | None = None
    dense_score: float = 0.0
    bm25_rank: int | None = None
    bm25_score: float = 0.0
    rrf_rank: int | None = None
    rerank_score: float | None = None


class Timings(BaseModel):
    rewrite_ms: float
    retrieval_ms: float
    generation_ms: float


class AgentStep(BaseModel):
    step: int
    thought: str
    action: str
    query: str | None = None
    chunks_found: int
    duration_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant_connected: bool
    collection_name: str
    collection_exists: bool
    points_count: int | None = None
    embedding_model: str
    rerank_model: str
    llm_configured: bool
    llm_model: str | None = None


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int = Field(default=3, ge=1, le=20)
    use_rerank: bool = False
    use_rewrite: bool = True
    use_guardrail: bool = True
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE
    min_dense_score: float = DEFAULT_MIN_DENSE_SCORE
    min_bm25_score: float = DEFAULT_MIN_BM25_SCORE
    use_agent: bool = False
    max_agent_steps: int = Field(default=3, ge=1, le=10)
    source_filter: list[str] | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Soru boş olamaz.")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("session_id boş olamaz.")
        if not SESSION_ID_PATTERN.match(cleaned):
            raise ValueError(
                "session_id en fazla 128 karakter olmalı ve yalnızca [A-Za-z0-9_-] içermelidir."
            )
        return cleaned

    @field_validator("source_filter")
    @classmethod
    def validate_source_filter(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]


class AskResponse(BaseModel):
    session_id: str
    answer: str
    rewritten_query: str
    early_rejected: bool
    rejection_reason: str | None = None
    chunks: list[ChunkResult]
    timings: Timings
    agent_steps: list[AgentStep] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=20)
    use_rerank: bool = False
    rrf_k: int = Field(default=10, ge=1)
    candidate_k: int = Field(default=6, ge=1, le=50)
    source_filter: list[str] | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Arama sorgusu boş olamaz.")
        return cleaned

    @field_validator("source_filter")
    @classmethod
    def validate_source_filter(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]


class SearchResponse(BaseModel):
    query: str
    chunks: list[ChunkResult]
    retrieval_ms: float


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionSummary(BaseModel):
    session_id: str
    turn_count: int
    max_turns: int
    created_at: float
    updated_at: float
    preview: str


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]
    total_sessions: int


class SessionResponse(BaseModel):
    session_id: str
    turn_count: int
    max_turns: int
    messages: list[SessionMessage]


class SessionDeleteResponse(BaseModel):
    session_id: str
    deleted: bool


class DocumentInfo(BaseModel):
    filename: str
    size_bytes: int
    extension: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DocumentUploadResponse(BaseModel):
    filename: str
    size_bytes: int
    chunk_count: int
    overwritten: bool
    message: str


class DocumentDeleteResponse(BaseModel):
    filename: str
    deleted_chunks: int
    deleted: bool


session_store = SqliteSessionStore(
    db_path=os.getenv("SESSION_DB_PATH", DEFAULT_SESSION_DB_PATH)
)

app = FastAPI(
    title="Doc-RAG Engine API",
    version="1.0.0",
    description="Hibrit arama, reranking, guardrail ve çok turlu oturum destekli RAG REST API.",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    qdrant_health = get_qdrant_health()
    api_key = (os.getenv("LLM_API_KEY") or "").strip()
    llm_model = (MODEL_NAME or os.getenv("LLM_MODEL_NAME") or "").strip() or None
    llm_configured = bool(api_key and llm_model)

    is_healthy = bool(
        qdrant_health.get("qdrant_connected") and qdrant_health.get("collection_exists")
    )
    status: Literal["ok", "degraded"] = "ok" if is_healthy else "degraded"

    return HealthResponse(
        status=status,
        qdrant_connected=bool(qdrant_health.get("qdrant_connected", False)),
        collection_name=str(qdrant_health.get("collection_name", "")),
        collection_exists=bool(qdrant_health.get("collection_exists", False)),
        points_count=qdrant_health.get("points_count"),
        embedding_model=EMBEDDING_MODEL,
        rerank_model=RERANK_MODEL,
        llm_configured=llm_configured,
        llm_model=llm_model,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    session_id = req.session_id or uuid.uuid4().hex
    history = session_store.get_history(session_id)

    try:
        ask_kwargs = {
            "question": req.question,
            "top_k": req.top_k,
            "use_rerank": req.use_rerank,
            "history": history,
            "use_rewrite": req.use_rewrite,
            "use_guardrail": req.use_guardrail,
            "min_rerank_score": req.min_rerank_score,
            "min_dense_score": req.min_dense_score,
            "min_bm25_score": req.min_bm25_score,
            "use_agent": req.use_agent,
            "max_agent_steps": req.max_agent_steps,
        }
        if req.source_filter is not None:
            ask_kwargs["source_filter"] = req.source_filter
        trace = ask_rag_with_trace(**ask_kwargs)
    except Exception:
        logger.exception("POST /ask sırasında beklenmeyen hata oluştu (session_id=%s)", session_id)
        raise HTTPException(
            status_code=502,
            detail="Yanıt üretilirken bir sunucu hatası oluştu.",
        )

    early_rejected = bool(trace.get("early_rejected", False))
    if not early_rejected:
        session_store.record_turn(session_id, req.question, trace["answer"])

    return AskResponse(
        session_id=session_id,
        answer=trace["answer"],
        rewritten_query=trace["rewritten_query"],
        early_rejected=early_rejected,
        rejection_reason=trace.get("rejection_reason"),
        chunks=[ChunkResult(**chunk) for chunk in trace.get("chunks", [])],
        timings=Timings(
            rewrite_ms=float(trace.get("rewrite_ms", 0.0)),
            retrieval_ms=float(trace.get("retrieval_ms", 0.0)),
            generation_ms=float(trace.get("generation_ms", 0.0)),
        ),
        agent_steps=[AgentStep(**s) for s in (trace.get("agent_steps") or [])],
    )


def _format_sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    session_id = req.session_id or uuid.uuid4().hex
    history = session_store.get_history(session_id)

    def event_generator() -> Iterator[str]:
        early_rejected = False
        rewrite_ms = 0.0
        retrieval_ms = 0.0

        try:
            stream_kwargs = {
                "question": req.question,
                "top_k": req.top_k,
                "use_rerank": req.use_rerank,
                "history": history,
                "use_rewrite": req.use_rewrite,
                "use_guardrail": req.use_guardrail,
                "min_rerank_score": req.min_rerank_score,
                "min_dense_score": req.min_dense_score,
                "min_bm25_score": req.min_bm25_score,
                "use_agent": req.use_agent,
                "max_agent_steps": req.max_agent_steps,
            }
            if req.source_filter is not None:
                stream_kwargs["source_filter"] = req.source_filter
            for item in ask_rag_stream(**stream_kwargs):
                event_type = item.get("type")
                if event_type == "meta":
                    early_rejected = bool(item.get("early_rejected", False))
                    rewrite_ms = float(item.get("rewrite_ms", 0.0))
                    retrieval_ms = float(item.get("retrieval_ms", 0.0))
                    meta_payload = {
                        "session_id": session_id,
                        "rewritten_query": item["rewritten_query"],
                        "early_rejected": early_rejected,
                        "rejection_reason": item.get("rejection_reason"),
                        "chunks": [
                            ChunkResult(**chunk).model_dump()
                            for chunk in item.get("chunks", [])
                        ],
                        "agent_steps": [
                            AgentStep(**step).model_dump()
                            for step in (item.get("agent_steps") or [])
                        ],
                        "timings": {
                            "rewrite_ms": rewrite_ms,
                            "retrieval_ms": retrieval_ms,
                            "generation_ms": 0.0,
                        },
                    }
                    yield _format_sse_event("meta", meta_payload)
                elif event_type == "token":
                    yield _format_sse_event("token", {"content": item["content"]})
                elif event_type == "done":
                    full_answer = item["answer"]
                    generation_ms = float(item.get("generation_ms", 0.0))
                    if not early_rejected:
                        session_store.record_turn(session_id, req.question, full_answer)
                    done_payload = {
                        "answer": full_answer,
                        "timings": {
                            "rewrite_ms": rewrite_ms,
                            "retrieval_ms": retrieval_ms,
                            "generation_ms": generation_ms,
                        },
                    }
                    yield _format_sse_event("done", done_payload)
        except Exception:
            logger.exception(
                "POST /ask/stream sırasında beklenmeyen hata oluştu (session_id=%s)",
                session_id,
            )
            yield _format_sse_event(
                "error",
                {"detail": "Yanıt üretilirken bir sunucu hatası oluştu."},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/search", response_model=SearchResponse)
def search_docs(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        search_kwargs = {
            "query": req.query,
            "top_k": req.top_k,
            "rrf_k": req.rrf_k,
            "use_rerank": req.use_rerank,
            "candidate_k": req.candidate_k,
        }
        if req.source_filter is not None:
            search_kwargs["source_filter"] = req.source_filter
        chunks = search(**search_kwargs)
    except Exception:
        logger.exception("POST /search sırasında beklenmeyen hata oluştu")
        raise HTTPException(
            status_code=503,
            detail="Arama servisi şu anda kullanılamıyor.",
        )

    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
    return SearchResponse(
        query=req.query,
        chunks=[ChunkResult(**chunk) for chunk in chunks],
        retrieval_ms=retrieval_ms,
    )


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions(limit: int = Query(50, ge=1, le=200)) -> SessionListResponse:
    raw_sessions = session_store.list_sessions(limit=limit)
    sessions = [SessionSummary(**item) for item in raw_sessions]
    return SessionListResponse(
        sessions=sessions,
        total_sessions=len(sessions),
    )


@app.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session_data = session_store.get_session(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
    return SessionResponse(**session_data)


@app.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
def delete_session(session_id: str) -> SessionDeleteResponse:
    deleted = session_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Oturum bulunamadı.")
    return SessionDeleteResponse(session_id=session_id, deleted=True)


@app.get("/documents", response_model=DocumentListResponse)
def get_documents() -> DocumentListResponse:
    try:
        raw_docs = indexer_list_documents()
    except Exception:
        logger.exception("GET /documents sırasında beklenmeyen hata oluştu")
        raise HTTPException(
            status_code=503,
            detail="Doküman listesi şu anda alınamıyor.",
        )

    documents = [DocumentInfo(**doc) for doc in raw_docs]
    total_chunks = sum(doc.chunk_count for doc in documents)
    return DocumentListResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks=total_chunks,
    )


@app.post("/documents", response_model=DocumentUploadResponse, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    overwrite: bool = Query(False),
) -> DocumentUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Dosya adı boş olamaz.")

    try:
        safe_name = validate_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    chunks_buf = bytearray()
    max_read = MAX_UPLOAD_SIZE_BYTES + 1
    try:
        while True:
            piece = file.file.read(65536)
            if not piece:
                break
            chunks_buf.extend(piece)
            if len(chunks_buf) >= max_read:
                raise HTTPException(
                    status_code=413,
                    detail=f"Dosya boyutu en fazla {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB olabilir.",
                )
    finally:
        file.file.close()

    content = bytes(chunks_buf)

    try:
        result = save_and_index_document(
            filename=safe_name,
            content=content,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except OverflowError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("POST /documents sırasında beklenmeyen hata oluştu")
        raise HTTPException(
            status_code=503,
            detail="Doküman indekslenirken bir sunucu hatası oluştu.",
        )

    action_text = "güncellendi" if result["overwritten"] else "yüklendi"
    message = (
        f"'{result['filename']}' {action_text} ve "
        f"{result['chunk_count']} parça olarak indekslendi."
    )
    return DocumentUploadResponse(
        filename=result["filename"],
        size_bytes=result["size_bytes"],
        chunk_count=result["chunk_count"],
        overwritten=result["overwritten"],
        message=message,
    )


@app.delete("/documents/{filename:path}", response_model=DocumentDeleteResponse)
def remove_document(filename: str) -> DocumentDeleteResponse:
    try:
        result = indexer_delete_document(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        logger.exception("DELETE /documents sırasında beklenmeyen hata oluştu")
        raise HTTPException(
            status_code=503,
            detail="Doküman silinirken bir sunucu hatası oluştu.",
        )

    return DocumentDeleteResponse(**result)
