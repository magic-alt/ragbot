from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from services.api.app.api import app

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "public_api_v1_contract.json"


def main() -> None:
    expected = json.loads(CONTRACT.read_text(encoding="utf-8"))
    schema = app.openapi()
    errors: list[str] = []

    for operation in expected.get("operations", []):
        path = str(operation["path"])
        method = str(operation["method"]).lower()
        path_item = schema.get("paths", {}).get(path)
        if not isinstance(path_item, Mapping):
            errors.append(f"missing public API path: {path}")
            continue
        actual = path_item.get(method)
        if not isinstance(actual, Mapping):
            errors.append(f"missing public API operation: {method.upper()} {path}")
            continue

        parameters = list(actual.get("parameters") or [])
        _check_parameters(
            path,
            method,
            parameters,
            expected_names=operation.get("query_parameters") or [],
            location="query",
            errors=errors,
        )
        _check_parameters(
            path,
            method,
            parameters,
            expected_names=operation.get("path_parameters") or [],
            location="path",
            errors=errors,
        )

        request_required = list(operation.get("request_required") or [])
        if request_required:
            request_schema = _json_schema(
                schema,
                actual.get("requestBody", {})
                .get("content", {})
                .get("application/json", {})
                .get("schema", {}),
            )
            actual_required = set(request_schema.get("required") or [])
            missing = sorted(set(request_required) - actual_required)
            if missing:
                errors.append(
                    f"{method.upper()} {path} removed required request fields: {missing}"
                )

        response_required = list(operation.get("response_required") or [])
        if response_required:
            responses = actual.get("responses", {})
            success_code = next(
                (
                    code
                    for code in ("200", "201", "202")
                    if isinstance(responses.get(code), Mapping)
                ),
                None,
            )
            if success_code is None:
                errors.append(f"{method.upper()} {path} has no declared success response")
                continue
            response_schema = _json_schema(
                schema,
                responses[success_code]
                .get("content", {})
                .get("application/json", {})
                .get("schema", {}),
            )
            actual_required = set(response_schema.get("required") or [])
            missing = sorted(set(response_required) - actual_required)
            if missing:
                errors.append(
                    f"{method.upper()} {path} removed required response fields: {missing}"
                )

    if errors:
        raise SystemExit("Public API compatibility gate failed:\n- " + "\n- ".join(errors))
    print(
        f"public API v{expected.get('schema_version')} compatibility OK: "
        f"{len(expected.get('operations') or [])} frozen operations"
    )


def _check_parameters(
    path: str,
    method: str,
    parameters: list[Any],
    *,
    expected_names: list[str],
    location: str,
    errors: list[str],
) -> None:
    actual = {
        str(item.get("name"))
        for item in parameters
        if isinstance(item, Mapping) and item.get("in") == location
    }
    missing = sorted(set(expected_names) - actual)
    if missing:
        errors.append(
            f"{method.upper()} {path} removed {location} parameters: {missing}"
        )


def _json_schema(openapi: Mapping[str, Any], value: Any) -> Mapping[str, Any]:
    current = value if isinstance(value, Mapping) else {}
    seen: set[str] = set()
    while "$ref" in current:
        ref = str(current["$ref"])
        if ref in seen:
            raise RuntimeError(f"cyclic OpenAPI reference: {ref}")
        seen.add(ref)
        if not ref.startswith("#/components/schemas/"):
            raise RuntimeError(f"unsupported OpenAPI reference: {ref}")
        name = ref.rsplit("/", 1)[-1]
        resolved = openapi.get("components", {}).get("schemas", {}).get(name)
        if not isinstance(resolved, Mapping):
            raise RuntimeError(f"unresolved OpenAPI schema: {ref}")
        current = resolved
    return current


if __name__ == "__main__":
    main()
