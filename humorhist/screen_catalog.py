"""Anonymous, fail-closed cost preflight for an exact Nous model ID.

This is only a catalog check, not a guarantee of account billing or access.
Screening uses it to check the exact model before requesting scores.
"""

import json
import re
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Literal

import httpx

_CATALOG_URL = "https://inference-api.nousresearch.com/v1/models"
_MAX_BYTES = 1_000_000
_TIMEOUT = 5.0
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}\Z", re.ASCII)
_DECIMAL = re.compile(r"[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z", re.ASCII)
_SAFE_PRICE_FIELDS = ("prompt", "completion", "input_cache_read")
Category = Literal["free", "invalid_model", "unavailable", "invalid_catalog", "model_missing", "unsafe_pricing"]
_MESSAGES = {
    "free": "Catalog reports zero prices for this model; billing and access are not guaranteed.",
    "invalid_model": "Invalid model identifier.",
    "unavailable": "Catalog could not be checked safely.",
    "invalid_catalog": "Catalog response was invalid.",
    "model_missing": "Exact model was not found in the catalog.",
    "unsafe_pricing": "Model pricing is missing, invalid, or not zero.",
}


@dataclass(frozen=True)
class CatalogResult:
    allowed: bool
    category: Category
    model_id: str | None
    prices: dict[str, str]
    message: str


def _result(category: Category, model_id: str | None, prices: dict[str, str] | None = None) -> CatalogResult:
    return CatalogResult(category == "free", category, model_id, prices or {}, _MESSAGES[category])


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(_value: str) -> None:
    raise ValueError("non-finite JSON number")


def _zero(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        return False
    if isinstance(value, str) and (len(value) > 128 or not _DECIMAL.fullmatch(value)):
        return False
    try:
        number = Decimal(value)
        return number.is_finite() and not number.is_signed() and number == 0
    except (InvalidOperation, ValueError, OverflowError):
        return False


def check_screen_model(model_id: str) -> CatalogResult:
    """Check an exact ID with one anonymous GET; never authorize nonzero costs.

    Network inactivity is bounded by HTTPX timeouts. A monotonic deadline is
    also checked as response chunks arrive, to reject slow streaming responses.
    No environment credentials, proxy configuration, redirects or retries are used.
    """
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        return _result("invalid_model", None)
    try:
        deadline = time.monotonic() + _TIMEOUT
        with (
            httpx.Client(trust_env=False, follow_redirects=False, timeout=_TIMEOUT) as client,
            client.stream("GET", _CATALOG_URL, headers={"Accept-Encoding": "identity"}) as response,
        ):
            if response.status_code != 200:
                return _result("unavailable", model_id)
            encoding = response.headers.get("content-encoding", "identity").lower()
            if encoding != "identity":
                return _result("invalid_catalog", model_id)
            body = bytearray()
            for chunk in response.iter_raw():
                if time.monotonic() >= deadline or len(body) + len(chunk) > _MAX_BYTES:
                    return _result("unavailable", model_id)
                body.extend(chunk)
            if time.monotonic() >= deadline:
                return _result("unavailable", model_id)
        try:
            catalog = json.loads(
                body, object_pairs_hook=_unique_object, parse_int=Decimal, parse_float=Decimal,
                parse_constant=_invalid_constant,
            )
        except (ValueError, UnicodeError, RecursionError):
            return _result("invalid_catalog", model_id)
        if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
            return _result("invalid_catalog", model_id)
        data = catalog["data"]
        if any(
            not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip()
            for item in data
        ):
            return _result("invalid_catalog", model_id)
        matches = [item for item in data if item["id"] == model_id]
        if not matches:
            return _result("model_missing", model_id)
        if len(matches) != 1:
            return _result("invalid_catalog", model_id)
        pricing = matches[0].get("pricing")
        if not isinstance(pricing, dict) or not {"prompt", "completion"} <= pricing.keys():
            return _result("unsafe_pricing", model_id)
        if not all(_zero(value) for value in pricing.values()):
            return _result("unsafe_pricing", model_id)
        # Do not echo untrusted provider field names or values.
        prices = {key: "0" for key in _SAFE_PRICE_FIELDS if key in pricing}
        return _result("free", model_id, prices)
    except Exception:
        # Never expose provider bodies, URLs, headers or exception text.
        return _result("unavailable", model_id)
