from __future__ import annotations

from env.models import Reward


def build_reward(
    *,
    classification_correct: bool = False,
    priority_correct: bool = False,
    action_correct: bool = False,
    folder_correct: bool = False,
    invalid_action: bool = False,
    repeated_action: bool = False,
    extra_step: bool = False,
    reason: str,
) -> Reward:
    """Build a deterministic dense reward for the current step."""

    breakdown: dict[str, float] = {}

    if classification_correct:
        breakdown["classification"] = 0.30
    if priority_correct:
        breakdown["priority"] = 0.30
    if action_correct:
        breakdown["workflow_action"] = 0.25
    if folder_correct:
        breakdown["folder"] = 0.15
    if invalid_action:
        breakdown["invalid_action"] = -0.10
    if repeated_action:
        breakdown["repeated_action"] = -0.10
    if extra_step:
        breakdown["extra_step_penalty"] = -0.05

    score = max(-1.0, min(1.0, sum(breakdown.values())))
    return Reward(score=score, reason=reason, breakdown=breakdown)
