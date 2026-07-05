"""Opt-in RAG-Anything bridge routes.

RAG-Anything is vendored in this fork instead of installed from PyPI because
raganything==1.3.1 can resolve LightRAG dependencies differently from this
server's pinned runtime. Keep all raganything imports lazy and behind
RAGANYTHING_ENABLE so the normal LightRAG API keeps working even if an optional
multimodal dependency is missing.
"""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from lightrag import LightRAG
from lightrag.utils import logger

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def raganything_enabled() -> bool:
    """Whether the RAG-Anything bridge is enabled for this process."""

    return _env_bool("RAGANYTHING_ENABLE", False)


@lru_cache(maxsize=1)
def _vendor_version() -> str | None:
    try:
        from raganything import get_version  # type: ignore

        return get_version()
    except Exception:
        try:
            from raganything import __version__  # type: ignore

            return str(__version__)
        except Exception:
            return None


class RagAnythingInsertRequest(BaseModel):
    """Direct content-list insertion request.

    Content items follow RAG-Anything's native schema, e.g.:
    - {"type":"text", "text":"...", "page_idx":0}
    - {"type":"image", "img_path":"/absolute/path.png", "image_caption":["..."], "page_idx":1}
    - {"type":"table", "table_body":"| A | B |", "page_idx":2}
    - {"type":"equation", "latex":"E=mc^2", "text":"...", "page_idx":3}
    """

    content_list: List[dict[str, Any]] = Field(..., min_length=1)
    file_path: str = "raganything_content_list"
    doc_id: Optional[str] = None
    split_by_character: Optional[str] = None
    split_by_character_only: bool = False
    display_stats: Optional[bool] = None


class RagAnythingQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = "mix"
    system_prompt: Optional[str] = None
    multimodal_content: Optional[List[dict[str, Any]]] = None
    vlm_enhanced: Optional[bool] = None
    top_k: Optional[int] = None
    response_type: Optional[str] = None
    only_need_context: Optional[bool] = None
    only_need_prompt: Optional[bool] = None


class RagAnythingProcessFileResponse(BaseModel):
    status: str
    file_name: str
    doc_id: Optional[str]


def _build_config():
    from raganything import RAGAnythingConfig  # type: ignore

    return RAGAnythingConfig(
        working_dir=os.getenv(
            "RAGANYTHING_WORKING_DIR", os.getenv("WORKING_DIR", "/app/data/rag_storage")
        ),
        parser=os.getenv("RAGANYTHING_PARSER", os.getenv("PARSER", "mineru")),
        parser_output_dir=os.getenv(
            "RAGANYTHING_OUTPUT_DIR",
            os.getenv("OUTPUT_DIR", "/app/data/rag_storage/raganything_output"),
        ),
        parse_method=os.getenv("RAGANYTHING_PARSE_METHOD", os.getenv("PARSE_METHOD", "auto")),
        enable_image_processing=_env_bool("RAGANYTHING_IMAGE_PROCESSING", True),
        enable_table_processing=_env_bool("RAGANYTHING_TABLE_PROCESSING", True),
        enable_equation_processing=_env_bool("RAGANYTHING_EQUATION_PROCESSING", True),
        display_content_stats=_env_bool("RAGANYTHING_DISPLAY_STATS", True),
    )


def _build_raganything(rag: LightRAG):
    """Create a RAG-Anything wrapper lazily for the current request.

    LightRAG's server config exposes VLM through role/env configuration, not as
    a stable ``rag.vlm_model_func`` attribute. Do not fake a vision function
    here. RAG-Anything will inherit the existing LightRAG LLM/embedding funcs;
    deployments that need richer image captioning should enable the native VLM
    role envs separately.
    """

    from raganything import RAGAnything  # type: ignore

    config = _build_config()
    return RAGAnything(
        lightrag=rag,
        config=config,
    )


def _disabled_payload() -> dict[str, Any]:
    return {
        "enabled": False,
        "reason": "RAGANYTHING_ENABLE is not true",
        "vendorized": True,
        "version": _vendor_version(),
    }


def create_raganything_routes(rag: LightRAG, auth_dependency) -> APIRouter:
    """Create opt-in RAG-Anything bridge routes."""

    router = APIRouter(prefix="/raganything", tags=["raganything"])

    @router.get("/health", dependencies=[Depends(auth_dependency)])
    async def raganything_health():
        if not raganything_enabled():
            return _disabled_payload()

        try:
            config = _build_config()
            version = _vendor_version()
            return {
                "enabled": True,
                "vendorized": True,
                "version": version,
                "config": {
                    "working_dir": config.working_dir,
                    "parser": config.parser,
                    "parser_output_dir": config.parser_output_dir,
                    "parse_method": config.parse_method,
                    "image_processing": config.enable_image_processing,
                    "table_processing": config.enable_table_processing,
                    "equation_processing": config.enable_equation_processing,
                },
            }
        except Exception as exc:
            logger.exception("RAG-Anything health check failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/insert-content-list", dependencies=[Depends(auth_dependency)])
    async def raganything_insert_content_list(payload: RagAnythingInsertRequest):
        if not raganything_enabled():
            raise HTTPException(status_code=404, detail=_disabled_payload())

        try:
            wrapper = _build_raganything(rag)
            config = _build_config()
            await wrapper.insert_content_list(
                content_list=payload.content_list,
                file_path=payload.file_path,
                split_by_character=payload.split_by_character,
                split_by_character_only=payload.split_by_character_only,
                doc_id=payload.doc_id,
                display_stats=payload.display_stats
                if payload.display_stats is not None
                else config.display_content_stats,
            )
            return {
                "status": "ok",
                "doc_id": payload.doc_id,
                "file_path": payload.file_path,
                "items": len(payload.content_list),
            }
        except Exception as exc:
            logger.exception("RAG-Anything content-list insertion failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post(
        "/process-file",
        dependencies=[Depends(auth_dependency)],
        response_model=RagAnythingProcessFileResponse,
    )
    async def raganything_process_file(
        file: UploadFile = File(...),
        doc_id: Optional[str] = Form(default=None),
        parse_method: Optional[str] = Form(default=None),
        display_stats: Optional[bool] = Form(default=None),
        split_by_character: Optional[str] = Form(default=None),
        split_by_character_only: bool = Form(default=False),
    ):
        if not raganything_enabled():
            raise HTTPException(status_code=404, detail=_disabled_payload())

        suffix = os.path.splitext(file.filename or "upload")[1]
        tmp_path: str | None = None
        try:
            wrapper = _build_raganything(rag)
            config = _build_config()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                while chunk := await file.read(1024 * 1024):
                    tmp.write(chunk)

            process_kwargs: dict[str, Any] = {
                "display_stats": display_stats
                if display_stats is not None
                else config.display_content_stats,
                "split_by_character": split_by_character,
                "split_by_character_only": split_by_character_only,
                "doc_id": doc_id,
                "file_name": file.filename,
            }
            if parse_method is not None:
                process_kwargs["parse_method"] = parse_method
            await wrapper.process_document_complete(tmp_path, **process_kwargs)
            return RagAnythingProcessFileResponse(
                status="ok", file_name=file.filename or "upload", doc_id=doc_id
            )
        except Exception as exc:
            logger.exception("RAG-Anything file processing failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @router.post("/query", dependencies=[Depends(auth_dependency)])
    async def raganything_query(payload: RagAnythingQueryRequest):
        if not raganything_enabled():
            raise HTTPException(status_code=404, detail=_disabled_payload())

        try:
            wrapper = _build_raganything(rag)
            kwargs: dict[str, Any] = {}
            for key in (
                "vlm_enhanced",
                "top_k",
                "response_type",
                "only_need_context",
                "only_need_prompt",
            ):
                value = getattr(payload, key)
                if value is not None:
                    kwargs[key] = value

            if payload.multimodal_content:
                result = await wrapper.aquery_with_multimodal(
                    payload.query,
                    multimodal_content=payload.multimodal_content,
                    mode=payload.mode,
                    system_prompt=payload.system_prompt,
                    **kwargs,
                )
            else:
                result = await wrapper.aquery(
                    payload.query,
                    mode=payload.mode,
                    system_prompt=payload.system_prompt,
                    **kwargs,
                )
            return {"status": "ok", "result": result}
        except Exception as exc:
            logger.exception("RAG-Anything query failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return router
