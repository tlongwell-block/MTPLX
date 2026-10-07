"""Brain sampling shared by Frankie's voice and HTTP paths."""

import math

from mtplx.sampling import SamplerConfig

THINKING_BUDGETS = {"off": 0, "minimal": 64, "low": 256, "medium": 1024,
                    "high": 4096, "xhigh": 16384, "max": 32768}


def thinking_mode(settings):
    effort = settings.get("reasoning_effort")
    mode = effort if effort is not None else settings.get("thinking", "off")
    if mode == "none":
        mode = "off"
    if mode not in THINKING_BUDGETS:
        raise ValueError("Unknown thinking level.")
    enabled = settings.get("enable_thinking")
    if enabled is not None:
        if type(enabled) is not bool:
            raise ValueError("enable_thinking must be boolean.")
        if effort is not None and enabled != (mode != "off"):
            raise ValueError("enable_thinking conflicts with reasoning_effort.")
        mode = (mode if mode != "off" else "high") if enabled else "off"
    return mode


def brain_sampler(settings, *, realtime=False):
    thinking = thinking_mode(settings) != "off"

    def number(key, default, low, high):
        value = settings.get(key)
        value = default if value is None else value
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"Invalid {key}.")
        return value

    top_k = number("top_k", 20, 0, 2**31 - 1)
    if type(top_k) is not int:
        raise ValueError("top_k must be an integer.")
    if settings.get("min_p") not in (None, 0):
        raise ValueError("Frankie currently supports min_p=0 only.")
    if any(settings.get(key) not in (None, 1) for key in ("repeat_penalty", "repetition_penalty")):
        raise ValueError("Frankie currently supports repeat_penalty=1 only.")
    return SamplerConfig(
        temperature=number("temperature", 1.0 if thinking else 0.7, 0, 2),
        top_p=number("top_p", 0.95 if thinking else 0.8, 0, 1), top_k=top_k,
        presence_penalty=number("presence_penalty", 0.0 if thinking or realtime else 1.5, -2, 2),
        frequency_penalty=number("frequency_penalty", 0.0, -2, 2),
    )


def thinking_guard(tokenizer, mode):
    from mtplx.thinking_guard import ThinkingGuardConfig, think_marker_ids

    markers = think_marker_ids(tokenizer)
    if mode == "off" or not markers:
        return None
    return ThinkingGuardConfig(enabled=True, think_open_token=markers[0],
        think_close_token=markers[1], budget_tokens=THINKING_BUDGETS[mode],
        forced_close_ids=(markers[1],), starts_in_think=True)
