from __future__ import annotations

from dataclasses import dataclass

from env.models import ProcessedEmail, TaskSpec


@dataclass(frozen=True)
class GradeResult:
    """Final deterministic score and component breakdown."""

    score: float
    breakdown: dict[str, float]


class Grader:
    """Deterministic grader for the email triage tasks."""

    def __init__(self, task: TaskSpec):
        self.task = task

    def score(self, processed_emails: list[ProcessedEmail], total_steps: int) -> GradeResult:
        total_items = len(self.task.expected_outcomes)
        if total_items == 0:
            return GradeResult(score=1.0, breakdown={"classification_accuracy": 1.0, "efficiency": 1.0})

        classification_hits = 0
        priority_hits = 0
        action_hits = 0
        folder_hits = 0

        for processed in processed_emails:
            truth = self.task.expected_outcomes[processed.email.id]
            decision = processed.decision

            if decision.label == truth.label:
                classification_hits += 1
            if truth.priority is not None and decision.priority == truth.priority:
                priority_hits += 1
            if truth.workflow_action is not None and decision.workflow_action == truth.workflow_action:
                action_hits += 1
            if truth.folder is not None and decision.folder == truth.folder:
                folder_hits += 1

        classification_accuracy = classification_hits / total_items
        breakdown: dict[str, float] = {"classification_accuracy": round(classification_accuracy, 4)}

        if self.task.task_level == "easy":
            core_score = classification_accuracy
        elif self.task.task_level == "medium":
            priority_accuracy = priority_hits / total_items
            breakdown["priority_accuracy"] = round(priority_accuracy, 4)
            core_score = (0.55 * classification_accuracy) + (0.45 * priority_accuracy)
        else:
            priority_accuracy = priority_hits / total_items
            action_accuracy = action_hits / total_items
            folder_accuracy = folder_hits / total_items
            breakdown["priority_accuracy"] = round(priority_accuracy, 4)
            breakdown["action_accuracy"] = round(action_accuracy, 4)
            breakdown["folder_accuracy"] = round(folder_accuracy, 4)
            core_score = (
                0.30 * classification_accuracy
                + 0.25 * priority_accuracy
                + 0.25 * action_accuracy
                + 0.10 * folder_accuracy
            )

        efficiency = max(0.0, 1.0 - max(0, total_steps - self.task.ideal_steps) / max(1, self.task.ideal_steps))
        breakdown["efficiency"] = round(efficiency, 4)
        final_score = max(0.0, min(1.0, (0.90 * core_score) + (0.10 * efficiency)))
        return GradeResult(score=round(final_score, 4), breakdown=breakdown)
