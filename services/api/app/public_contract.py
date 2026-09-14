from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

API_VERSION = "v1"
API_PREFIX = f"/{API_VERSION}"


def install_public_api_contract(app: Any) -> None:
    """Install stable error envelopes for versioned public endpoints only.

    Legacy endpoints keep FastAPI's historical `{\"detail\": ...}` shape. New
    `/v1` endpoints use a typed envelope consumed by maintained SDKs.
    """

    @app.exception_handler(HTTPException)
    async def public_http_exception(request: Request, exc: HTTPException):
        if not request.url.path.startswith(API_PREFIX):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
        code = _error_code(exc.status_code)
        detail = exc.detail
        if isinstance(detail, dict):
            message = str(detail.get("message") or detail.get("detail") or code)
            details = dict(detail)
            details.pop("message", None)
        else:
            message = str(detail)
            details = None
        return _error_response(
            request,
            status_code=exc.status_code,
            code=code,
            message=message,
            details=details,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def public_validation_exception(request: Request, exc: RequestValidationError):
        if not request.url.path.startswith(API_PREFIX):
            return JSONResponse(status_code=422, content={"detail": exc.errors()})
        return _error_response(
            request,
            status_code=422,
            code="validation_error",
            message="Request validation failed",
            details={"errors": exc.errors()},
        )


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Any = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    payload = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
            "retryable": status_code in {429, 502, 503, 504},
            "details": details,
        }
    }
    merged_headers = dict(headers or {})
    merged_headers["X-Ragbot-API-Version"] = API_VERSION
    if request_id:
        merged_headers.setdefault("X-Request-ID", str(request_id))
    return JSONResponse(status_code=status_code, content=payload, headers=merged_headers)


def _error_code(status_code: int) -> str:
    return {
        400: "invalid_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "rate_limited",
        502: "upstream_error",
        503: "service_unavailable",
        504: "deadline_exceeded",
    }.get(status_code, "http_error")
