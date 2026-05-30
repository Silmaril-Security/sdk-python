# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Core Silmaril Firewall client."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from typing import Any
from uuid import uuid4

import requests
from typing_extensions import deprecated

from silmaril_security.sdk._version import VERSION
from silmaril_security.sdk.chunking import SERVER_SINGLE_TEXT_MAX_CHARS, chunk_text
from silmaril_security.sdk.exceptions import (
    BatchFirewallBlockedException,
    FirewallBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk.hooks import HookLabel, hook_value, normalize_hook_label
from silmaril_security.sdk.types import (
    BlockedBatchItem,
    BlockResult,
    ClassificationMetadata,
    ClassifyEvent,
    Prediction,
)

LOG = logging.getLogger("silmaril_security.sdk")

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_CHUNK_CONCURRENCY = 8
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_BACKOFF_SECONDS = 30.0
_MAX_ERROR_BODY_BYTES = 1 << 16


@deprecated("Thresholds are tenant-owned by the Firewall backend.")
def adaptive_threshold(chunk_count: int) -> float:
    """Return the legacy adaptive threshold for a scoring-opportunity count."""
    import math

    if chunk_count < 1:
        raise ValueError(f"Firewall: chunk_count must be >= 1, got {chunk_count}")
    if chunk_count == 1:
        return 0.5
    target_chunk_fpr = 1.0 - math.pow(1.0 - 0.01, 1.0 / chunk_count)
    odds_ratio = 0.01 / target_chunk_fpr
    raw_threshold = odds_ratio / (1.0 + odds_ratio)
    return min(raw_threshold, 0.9)


def _parse_outcome_scores(data: dict[str, Any]) -> dict[str, float] | None:
    raw = data.get("outcome_scores")
    if raw is None:
        return None
    return {str(k): float(v) for k, v in raw.items() if v is not None}


def _prediction_for_score(score: float, threshold: float) -> Prediction:
    return "MALICIOUS" if score >= threshold else "BENIGN"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _score_from_json(data: dict[str, Any], *, prediction: Prediction, threshold: float) -> float:
    score = _optional_float(data.get("score"))
    if score is not None:
        return score
    return threshold if prediction == "MALICIOUS" else 0.0


def _block_result_from_json(data: dict[str, Any]) -> BlockResult:
    threshold = _optional_float(data.get("threshold")) or 0.5
    raw_prediction = data.get("prediction")
    score = _optional_float(data.get("score"))
    prediction = raw_prediction or (
        _prediction_for_score(score, threshold) if score is not None else None
    )
    if prediction not in ("BENIGN", "MALICIOUS"):
        raise ValueError(f"Firewall: invalid prediction {prediction!r}")
    score = _score_from_json(data, prediction=prediction, threshold=threshold)
    return BlockResult(
        prediction=prediction,
        score=score,
        threshold=threshold,
        primary_outcome=data.get("primary_outcome"),
        outcome_scores=_parse_outcome_scores(data),
    )


def _sdk_metadata(
    metadata: ClassificationMetadata | None,
    *,
    request_id: str,
    input_index: int,
    chunk_index: int,
    chunk_count: int,
) -> dict[str, Any]:
    payload = dict(metadata) if metadata is not None else {}
    existing = payload.get("silmaril")
    if existing is not None and not isinstance(existing, Mapping):
        raise ValueError("Firewall: metadata.silmaril must be an object when provided")
    namespace = dict(existing) if isinstance(existing, Mapping) else {}
    namespace.update(
        {
            "sdk_language": "python",
            "sdk_version": VERSION,
            "request_id": request_id,
            "input_index": input_index,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
        }
    )
    payload["silmaril"] = namespace
    return payload


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        seconds = int(stripped)
    except ValueError:
        try:
            when = parsedate_to_datetime(stripped)
        except (TypeError, ValueError):
            return None
        delay = when.timestamp() - time.time()
        return max(delay, 0.0)
    return float(seconds) if seconds >= 0 else None


def _read_capped_error_body(response: Any, limit: int = _MAX_ERROR_BODY_BYTES) -> str:
    try:
        if hasattr(response, "iter_content"):
            chunks: list[bytes] = []
            remaining = limit
            for chunk in response.iter_content(chunk_size=min(8192, limit)):
                if not chunk:
                    continue
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                if len(chunk) > remaining:
                    chunks.append(chunk[:remaining])
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
                if remaining <= 0:
                    break
            return b"".join(chunks).decode("utf-8", errors="replace")
        return str(response.text)[:limit]
    except Exception:
        return ""


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:
        pass


class Firewall:
    """Client for the Silmaril Firewall /classify endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        timeout: float = DEFAULT_TIMEOUT,
        shadow_mode: bool = False,
        on_classify: Callable[[ClassifyEvent], None] | None = None,
        session: requests.Session | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        chunk_concurrency: int = DEFAULT_CHUNK_CONCURRENCY,
    ) -> None:
        if not api_key:
            raise ValueError("Firewall: api_key is required")
        if not api_url:
            raise ValueError("Firewall: api_url is required")
        if timeout < 0:
            raise ValueError(f"Firewall: timeout must be non-negative, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"Firewall: max_retries must be non-negative, got {max_retries}")
        if chunk_concurrency < 1:
            raise ValueError(
                f"Firewall: chunk_concurrency must be >= 1, got {chunk_concurrency}"
            )

        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout
        self.shadow_mode = shadow_mode
        self.on_classify = on_classify
        self.max_retries = max_retries
        self.chunk_concurrency = chunk_concurrency
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "x-api-key": self.api_key,
                "content-type": "application/json",
            }
        )

    def classify(
        self,
        text: str,
        *,
        hook: HookLabel | str | None = None,
        tool_name: str | None = None,
        metadata: ClassificationMetadata | None = None,
        shadow_mode: bool | None = None,
        request_id: str | None = None,
    ) -> BlockResult:
        """Classify a single text and enforce the backend tenant threshold."""
        request_id_value = request_id or str(uuid4())
        result = self._classify_raw(
            text,
            hook=hook,
            tool_name=tool_name,
            metadata=metadata,
            request_id=request_id_value,
        )
        event = self._new_classify_event(
            text=text,
            hook=hook,
            tool_name=tool_name,
            result=result,
            shadow_mode=self._effective_shadow_mode(shadow_mode),
        )
        self._fire_on_classify(event)
        if event.blocked and not event.shadow_mode:
            raise FirewallBlockedException(
                score=result.score,
                threshold=result.threshold,
                prompt_text=text,
                hook=event.hook,
                tool_name=tool_name,
                result=result,
            )
        return result

    def classify_batch(
        self,
        texts: Sequence[str],
        *,
        hooks: Sequence[HookLabel | str] | None = None,
        tool_names: Sequence[str | None] | None = None,
        metadata: Sequence[ClassificationMetadata | None] | None = None,
        shadow_mode: bool | None = None,
        request_id: str | None = None,
    ) -> list[BlockResult]:
        """Classify multiple independent texts and enforce backend tenant thresholds."""
        request_id_value = request_id or str(uuid4())
        results = self._classify_batch_raw(
            texts,
            hooks=hooks,
            tool_names=tool_names,
            metadata=metadata,
            request_id=request_id_value,
        )
        effective_shadow = self._effective_shadow_mode(shadow_mode)
        blocked: list[BlockedBatchItem] = []
        for index, result in enumerate(results):
            hook = hooks[index] if hooks is not None else None
            tool_name = tool_names[index] if tool_names is not None else None
            event = self._new_classify_event(
                text=texts[index],
                hook=hook,
                tool_name=tool_name,
                result=result,
                shadow_mode=effective_shadow,
            )
            self._fire_on_classify(event)
            if event.blocked and not event.shadow_mode:
                blocked.append(
                    BlockedBatchItem(
                        index=index,
                        text=texts[index],
                        hook=event.hook,
                        tool_name=tool_name,
                        result=result,
                    )
                )
        if blocked:
            raise BatchFirewallBlockedException(blocked=blocked, results=results)
        return results

    def as_langchain_handler(self, **options: Any) -> Any:
        """Create a synchronous LangChain callback handler."""
        from silmaril_security.sdk.langchain import SilmarilFirewallHandler

        return SilmarilFirewallHandler(self, **options)

    def as_async_langchain_handler(self, **options: Any) -> Any:
        """Create an asynchronous LangChain callback handler."""
        from silmaril_security.sdk.langchain import AsyncSilmarilFirewallHandler

        return AsyncSilmarilFirewallHandler(self, **options)

    def _classify_raw(
        self,
        text: str,
        *,
        hook: HookLabel | str | None = None,
        tool_name: str | None = None,
        metadata: ClassificationMetadata | None = None,
        request_id: str,
    ) -> BlockResult:
        chunks = [text] if len(text) <= SERVER_SINGLE_TEXT_MAX_CHARS else chunk_text(text)
        if len(chunks) == 1:
            return self._classify_single_raw(
                chunks[0],
                hook=hook,
                tool_name=tool_name,
                metadata=_sdk_metadata(
                    metadata,
                    request_id=request_id,
                    input_index=0,
                    chunk_index=0,
                    chunk_count=1,
                ),
            )

        workers = min(self.chunk_concurrency, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda item: self._classify_single_raw(
                        item[1],
                        hook=hook,
                        tool_name=tool_name,
                        metadata=_sdk_metadata(
                            metadata,
                            request_id=request_id,
                            input_index=0,
                            chunk_index=item[0],
                            chunk_count=len(chunks),
                        ),
                    ),
                    enumerate(chunks),
                )
            )
        return max(results, key=lambda result: result.score)

    def _classify_single_raw(
        self,
        text: str,
        *,
        hook: HookLabel | str | None = None,
        tool_name: str | None = None,
        metadata: ClassificationMetadata | None = None,
    ) -> BlockResult:
        payload: dict[str, Any] = {"text": text}
        hook_str = hook_value(hook)
        if hook_str:
            payload["hook"] = hook_str
        if tool_name:
            payload["tool_name"] = tool_name
        if metadata is not None:
            payload["metadata"] = dict(metadata)
        data = self._post_json(payload)
        return _block_result_from_json(data)

    def _classify_batch_raw(
        self,
        texts: Sequence[str],
        *,
        hooks: Sequence[HookLabel | str] | None = None,
        tool_names: Sequence[str | None] | None = None,
        metadata: Sequence[ClassificationMetadata | None] | None = None,
        request_id: str,
    ) -> list[BlockResult]:
        text_list = list(texts)
        if not text_list:
            raise ValueError("Firewall: texts must not be empty")
        if hooks is not None and len(hooks) != len(text_list):
            raise ValueError(
                f"Firewall: hooks length {len(hooks)} does not match texts length {len(text_list)}"
            )
        if tool_names is not None and len(tool_names) != len(text_list):
            raise ValueError(
                "Firewall: tool_names length "
                f"{len(tool_names)} does not match texts length {len(text_list)}"
            )
        if metadata is not None and len(metadata) != len(text_list):
            raise ValueError(
                f"Firewall: metadata length {len(metadata)} does not match texts length "
                f"{len(text_list)}"
            )

        payload: dict[str, Any] = {"texts": text_list}
        if hooks:
            payload["hooks"] = [hook_value(h) for h in hooks]
        if tool_names:
            payload["tool_names"] = list(tool_names)
        payload["metadata"] = [
            _sdk_metadata(
                metadata[index] if metadata is not None else None,
                request_id=request_id,
                input_index=index,
                chunk_index=0,
                chunk_count=1,
            )
            for index in range(len(text_list))
        ]

        data = self._post_json(payload)
        predictions = data["predictions"]
        if len(predictions) != len(text_list):
            raise ValueError(
                "Firewall: predictions length "
                f"{len(predictions)} does not match texts length {len(text_list)}"
            )
        return [_block_result_from_json(item) for item in predictions]

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    self.api_url,
                    data=body,
                    timeout=self.timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException:
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                _close_response(response)
                self._sleep_before_retry(attempt, response)
                continue
            if response.status_code >= 300:
                error_body = _read_capped_error_body(response)
                _close_response(response)
                raise SilmarilApiError(
                    status=response.status_code,
                    status_text=getattr(response, "reason", "") or "",
                    body=error_body,
                )
            return response.json()

        raise RuntimeError("Firewall: exhausted retries")

    def _sleep_before_retry(self, attempt: int, response: requests.Response | None) -> None:
        retry_after = _retry_after_seconds(response.headers.get("Retry-After")) if response else None
        wait = retry_after if retry_after is not None else min(2**attempt, _MAX_BACKOFF_SECONDS)
        LOG.debug("retrying firewall request in %.2fs after attempt %d", wait, attempt + 1)
        time.sleep(wait)

    def _effective_shadow_mode(self, shadow_mode: bool | None) -> bool:
        return self.shadow_mode if shadow_mode is None else shadow_mode

    def _new_classify_event(
        self,
        *,
        text: str,
        hook: HookLabel | str | None,
        tool_name: str | None,
        result: BlockResult,
        shadow_mode: bool,
    ) -> ClassifyEvent:
        return ClassifyEvent(
            hook=normalize_hook_label(hook),
            tool_name=tool_name,
            text=text,
            result=result,
            blocked=result.score >= result.threshold,
            shadow_mode=shadow_mode,
        )

    def _fire_on_classify(self, event: ClassifyEvent) -> None:
        if self.on_classify is None:
            return
        try:
            self.on_classify(event)
        except Exception:
            LOG.warning("on_classify callback raised", exc_info=True)


SilmarilFirewall = Firewall
