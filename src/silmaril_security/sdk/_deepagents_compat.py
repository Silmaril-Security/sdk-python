"""Version-bounded compatibility with Deep Agents 0.7.x profile internals."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from importlib.metadata import version
from typing import Any

from packaging.version import Version

_MIN_DEEPAGENTS = Version("0.7.30")
_MAX_DEEPAGENTS = Version("0.8")
_GENERAL_PURPOSE_PROFILE_FIELDS = {"enabled", "description", "system_prompt"}
_GENERAL_PURPOSE_INHERITABLE_MIDDLEWARE = frozenset({
    "AnthropicPromptCachingMiddleware",
    "BedrockPromptCachingMiddleware",
    "FilesystemMiddleware",
    "ModelRefusalMiddleware",
    "PatchToolCallsMiddleware",
    "SkillsMiddleware",
    "SummarizationMiddleware",
})


class DeepAgentsCompatibilityError(RuntimeError):
    """Raised when the qualified Deep Agents private contract has changed."""


def _compatibility_error(detail: str) -> DeepAgentsCompatibilityError:
    return DeepAgentsCompatibilityError(
        "Silmaril's protected Deep Agents factory supports the profile contract in "
        f"deepagents>=0.7.30,<0.8; {detail}. Install a qualified 0.7.x release or "
        "upgrade silmaril-security-sdk for the new Deep Agents contract."
    )


def _has_non_gp_profile_customization(profile: Any) -> bool:
    """Return whether a profile needs upstream object-hint recovery for children."""
    return any((
        getattr(profile, "base_system_prompt", None) is not None,
        getattr(profile, "system_prompt_suffix", None) is not None,
        bool(getattr(profile, "tool_description_overrides", {})),
        bool(getattr(profile, "excluded_tools", ())),
        bool(getattr(profile, "excluded_middleware", ())),
        bool(getattr(profile, "extra_middleware", ())),
    ))


def resolve_model_and_profile(model: Any) -> tuple[Any, Any]:
    """Return the parent model argument and its released-profile selection."""
    installed = Version(version("deepagents"))
    if not _MIN_DEEPAGENTS <= installed < _MAX_DEEPAGENTS:
        raise _compatibility_error(f"found deepagents=={installed}")

    try:
        from deepagents import HarnessProfile
        from deepagents.graph import _build_default_model, warn_deprecated
        from deepagents.profiles import _builtin_profiles
        from deepagents.profiles.harness.harness_profiles import (
            _HARNESS_PROFILES,
            _get_harness_profile,
            _harness_profile_for_model,
        )
    except (ImportError, AttributeError) as exc:  # pragma: no cover - version guard
        raise _compatibility_error(
            "required model/profile resolution helpers are unavailable"
        ) from exc

    try:
        if isinstance(model, str):
            # Keep the spec for create_deep_agent so upstream constructs the model
            # exactly once. Profile lookup from a spec does not need a model object.
            profile = _get_harness_profile(model) or HarnessProfile()
            provider, separator, identifier = model.partition(":")
            matching_keys = {model}
            if separator and provider and identifier:
                matching_keys.add(provider)
            custom_keys = {
                key
                for key in matching_keys
                if key in _HARNESS_PROFILES
                and key not in _builtin_profiles._BOOTSTRAP_HARNESS_KEYS
            }
            if custom_keys and _has_non_gp_profile_customization(profile):
                raise _compatibility_error(
                    "a runtime-registered string profile selected by "
                    f"{sorted(custom_keys)!r} contains root-level prompt, tool, or "
                    "middleware customizations. The released API cannot prove that "
                    "the once-resolved model will recover those keys for the explicit "
                    "protected child. Use a pre-built model whose provider/identifier "
                    "selects the profile, or limit this string-only profile to "
                    "general_purpose_subagent enabled/description/system_prompt"
                )
            return model, profile
        if model is None:
            warn_deprecated(
                since="0.5.3",
                removal="1.0.0",
                message=(
                    "Passing `model=None` to `create_deep_agent` is deprecated "
                    "and will be removed in deepagents==1.0.0. The `model` "
                    "parameter type will change from `BaseChatModel | str | None` "
                    "to `BaseChatModel | str`. Specify a model explicitly "
                    "(e.g., `ChatAnthropic(model_name=...)`). See "
                    "https://docs.langchain.com/oss/python/deepagents/models"
                ),
                package="deepagents",
            )
            actual_model = _build_default_model()
        else:
            actual_model = model
        profile = _harness_profile_for_model(actual_model, None)
    except (AttributeError, TypeError) as exc:
        raise _compatibility_error("model/profile resolution helpers changed shape") from exc

    return actual_model, profile


def protected_default_subagent(
    profile: Any,
    middleware: Any,
    *,
    has_explicit_default: bool,
    root_middleware: list[Any],
    tools: Any,
    skills: Any,
) -> dict[str, Any] | None:
    """Build only the auto general-purpose spec while preserving profile intent."""
    if has_explicit_default:
        return None

    try:
        from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
    except (ImportError, AttributeError) as exc:  # pragma: no cover - version guard
        raise _compatibility_error("the general-purpose subagent template is unavailable") from exc

    required_profile_attributes = (
        "base_system_prompt",
        "system_prompt_suffix",
        "general_purpose_subagent",
        "materialize_extra_middleware",
    )
    if any(not hasattr(profile, name) for name in required_profile_attributes):
        raise _compatibility_error("the resolved HarnessProfile has an unknown shape")
    if not callable(profile.materialize_extra_middleware):
        raise _compatibility_error("HarnessProfile.materialize_extra_middleware is not callable")

    gp_profile = profile.general_purpose_subagent
    gp_description = None
    gp_prompt = None
    if gp_profile is not None:
        if not is_dataclass(gp_profile):
            raise _compatibility_error("GeneralPurposeSubagentProfile is no longer a dataclass")
        field_names = {field.name for field in fields(gp_profile)}
        if field_names != _GENERAL_PURPOSE_PROFILE_FIELDS:
            raise _compatibility_error(
                "GeneralPurposeSubagentProfile fields changed from "
                "enabled/description/system_prompt"
            )
        if getattr(gp_profile, "enabled") is False:
            return None
        gp_description = getattr(gp_profile, "description")
        gp_prompt = getattr(gp_profile, "system_prompt")

    if not isinstance(GENERAL_PURPOSE_SUBAGENT, dict) or not {
        "name",
        "description",
        "system_prompt",
    }.issubset(GENERAL_PURPOSE_SUBAGENT):
        raise _compatibility_error("the general-purpose subagent template changed shape")

    if gp_prompt is not None and profile.base_system_prompt is not None:
        raise _compatibility_error(
            "a profile combines general_purpose_subagent.system_prompt with "
            "base_system_prompt, whose precedence cannot be preserved through "
            "the released declarative subagent API"
        )

    profile_extra = getattr(profile, "extra_middleware", ())
    if callable(profile_extra) and root_middleware:
        raise _compatibility_error(
            "the profile uses an extra_middleware factory while the root agent "
            "supplies custom middleware; the released API does not expose enough "
            "information to preserve auto-default middleware override inheritance "
            "without invoking that factory an extra time"
        )
    try:
        profile_extra_names = (
            set() if callable(profile_extra) else {entry.name for entry in profile_extra}
        )
        inherited = [
            entry
            for entry in root_middleware
            if entry.name
            in (_GENERAL_PURPOSE_INHERITABLE_MIDDLEWARE | profile_extra_names)
        ]
    except (AttributeError, TypeError) as exc:
        raise _compatibility_error(
            "profile or root middleware entries no longer expose the expected name"
        ) from exc

    # Deep Agents applies the active profile to this explicit declarative spec.
    # Passing the unmodified stock/custom base lets upstream apply base/suffix
    # prompts, tool rewrites, exclusions and profile middleware exactly once.
    protected = {
        **GENERAL_PURPOSE_SUBAGENT,
        "tools": tools,
        "middleware": [*inherited, middleware],
    }
    if skills is not None:
        protected["skills"] = skills
    if gp_description is not None:
        protected["description"] = gp_description
    if gp_prompt is not None:
        protected["system_prompt"] = gp_prompt
    return protected
