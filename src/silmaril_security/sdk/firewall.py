# Copyright (c) 2024-2026 Silmaril Security Inc. All rights reserved.

"""Core Silmaril Firewall client."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from typing import Any

import requests

from silmaril_security.sdk.chunking import chunk_text
from silmaril_security.sdk.exceptions import (
    BatchPromptBlockedException,
    PromptBlockedException,
    SilmarilApiError,
)
from silmaril_security.sdk.hooks import HookLabel, hook_value, normalize_hook_label
from silmaril_security.sdk.types import (
    BlockedBatchItem,
    BlockResult,
    ClassifyEvent,
    ExplainResult,
    Prediction,
)

LOG = logging.getLogger("silmaril_security.sdk")

DEFAULT_THRESHOLD = 0.5
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_CHUNK_CONCURRENCY = 8
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_MAX_BACKOFF_SECONDS = 30.0


def _validate_threshold(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Firewall: {name} must be between 0 and 1, got {value}")


def _parse_outcome_scores(data: dict[str, Any]) -> dict[str, float] | None:
    raw = data.get("outcome_scores")
    if raw is None:
        return None
    return {str(k): float(v) for k, v in raw.items()}


def _prediction_for_score(score: float, threshold: float) -> Prediction:
    return "MALICIOUS" if score >= threshold else "BENIGN"


def _block_result_from_json(data: dict[str, Any], threshold: float) -> BlockResult:
    score = float(data["score"])
    prediction = data.get("prediction") or _prediction_for_score(score, threshold)
    if prediction not in ("BENIGN", "MALICIOUS"):
        raise ValueError(f"Firewall: invalid prediction {prediction!r}")
    return BlockResult(
        prediction=prediction,
        score=score,
        threshold=threshold,
        primary_outcome=data.get("primary_outcome"),
        outcome_scores=_parse_outcome_scores(data),
    )


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


class Firewall:
    """Client for the Silmaril Firewall /classify endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        threshold: float = DEFAULT_THRESHOLD,
        timeout: float = DEFAULT_TIMEOUT,
        hook_thresholds: dict[HookLabel | str, float] | None = None,
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
        _validate_threshold("threshold", threshold)
        if timeout < 0:
            raise ValueError(f"Firewall: timeout must be non-negative, got {timeout}")
        if max_retries < 0:
            raise ValueError(f"Firewall: max_retries must be non-negative, got {max_retries}")
        if chunk_concurrency < 1:
            raise ValueError(
                f"Firewall: chunk_concurrency must be >= 1, got {chunk_concurrency}"
            )

        normalized_hook_thresholds: dict[str, float] = {}
        for hook, hook_threshold in (hook_thresholds or {}).items():
            _validate_threshold(f"hook_thresholds[{hook!r}]", hook_threshold)
            value = hook_value(hook)
            if value is None:
                continue
            normalized_hook_thresholds[value] = hook_threshold

        self.api_key = api_key
        self.api_url = api_url
        self.threshold = threshold
        self.timeout = timeout
        self.hook_thresholds = normalized_hook_thresholds
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
        shadow_mode: bool | None = None,
    ) -> BlockResult:
        """Classify a single text and enforce the effective threshold."""
        threshold = self.effective_threshold(hook)
        result = self._classify_raw(text, hook=hook, tool_name=tool_name, threshold=threshold)
        event = self._new_classify_event(
            text=text,
            hook=hook,
            tool_name=tool_name,
            result=result,
            shadow_mode=self._effective_shadow_mode(shadow_mode),
        )
        self._fire_on_classify(event)
        if event.blocked and not event.shadow_mode:
            raise PromptBlockedException(
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
        threshold: float | None = None,
        shadow_mode: bool | None = None,
    ) -> list[BlockResult]:
        """Classify multiple independent texts and enforce thresholds."""
        threshold_value = self._batch_threshold(hooks, threshold)
        results = self._classify_batch_raw(
            texts,
            hooks=hooks,
            tool_names=tool_names,
            threshold=threshold_value,
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
            raise BatchPromptBlockedException(blocked=blocked, results=results)
        return results

    def explain(
        self,
        text: str,
        *,
        hook: HookLabel | str | None = None,
        tool_name: str | None = None,
        steps: int = 50,
        temperature: float | None = None,
    ) -> ExplainResult:
        """Compute token-level attributions for a single text."""
        payload: dict[str, Any] = {"explain": True, "text": text}
        hook_str = hook_value(hook)
        if hook_str:
            payload["hook"] = hook_str
        if tool_name:
            payload["tool_name"] = tool_name
        if steps != 50:
            payload["steps"] = steps
        if temperature is not None:
            payload["temperature"] = temperature
        data = self._post_json(payload)
        prediction = data["prediction"]
        if prediction not in ("BENIGN", "MALICIOUS"):
            raise ValueError(f"Firewall: invalid prediction {prediction!r}")
        return ExplainResult(
            tokens=data["tokens"],
            attributions=data["attributions"],
            score=float(data["score"]),
            prediction=prediction,
            prepared_text=data["prepared_text"],
            primary_outcome=data.get("primary_outcome"),
            outcome_scores=_parse_outcome_scores(data),
        )

    def effective_threshold(self, hook: HookLabel | str | None = None) -> float:
        """Return the threshold that applies to a hook."""
        hook_str = hook_value(hook)
        if hook_str and hook_str in self.hook_thresholds:
            return self.hook_thresholds[hook_str]
        return self.threshold

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
        threshold: float | None = None,
    ) -> BlockResult:
        threshold_value = self.effective_threshold(hook) if threshold is None else threshold
        _validate_threshold("threshold", threshold_value)
        chunks = chunk_text(text)
        if len(chunks) == 1:
            return self._classify_single_raw(
                chunks[0],
                hook=hook,
                tool_name=tool_name,
                threshold=threshold_value,
            )

        workers = min(self.chunk_concurrency, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda chunk: self._classify_single_raw(
                        chunk,
                        hook=hook,
                        tool_name=tool_name,
                        threshold=threshold_value,
                    ),
                    chunks,
                )
            )
        return max(results, key=lambda result: result.score)

    def _classify_single_raw(
        self,
        text: str,
        *,
        hook: HookLabel | str | None = None,
        tool_name: str | None = None,
        threshold: float,
    ) -> BlockResult:
        payload: dict[str, Any] = {"text": text, "threshold": threshold}
        hook_str = hook_value(hook)
        if hook_str:
            payload["hook"] = hook_str
        if tool_name:
            payload["tool_name"] = tool_name
        data = self._post_json(payload)
        return _block_result_from_json(data, threshold)

    def _classify_batch_raw(
        self,
        texts: Sequence[str],
        *,
        hooks: Sequence[HookLabel | str] | None = None,
        tool_names: Sequence[str | None] | None = None,
        threshold: float | None = None,
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

        threshold_value = self._batch_threshold(hooks, threshold)
        payload: dict[str, Any] = {"texts": text_list, "threshold": threshold_value}
        if hooks:
            payload["hooks"] = [hook_value(h) for h in hooks]
        if tool_names:
            payload["tool_names"] = list(tool_names)

        data = self._post_json(payload)
        predictions = data["predictions"]
        if len(predictions) != len(text_list):
            raise ValueError(
                "Firewall: predictions length "
                f"{len(predictions)} does not match texts length {len(text_list)}"
            )
        return [_block_result_from_json(item, threshold_value) for item in predictions]

    def _batch_threshold(
        self,
        hooks: Sequence[HookLabel | str] | None,
        threshold: float | None,
    ) -> float:
        if threshold is not None:
            _validate_threshold("batch threshold", threshold)
            return threshold
        if not hooks:
            return self.threshold
        first = hook_value(hooks[0])
        if all(hook_value(hook) == first for hook in hooks):
            return self.effective_threshold(first)
        return self.threshold

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    self.api_url,
                    data=body,
                    timeout=self.timeout,
                )
            except requests.RequestException:
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt, None)
                    continue
                raise

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                self._sleep_before_retry(attempt, response)
                continue
            if response.status_code >= 400:
                try:
                    error_body = response.text
                except Exception:
                    error_body = ""
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
