from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import ValidationError

from env.graders import Grader
from env.models import Action, Observation, ProcessedEmail, Reward, TaskLevel, TaskSpec
from env.reward import build_reward
from env.tasks import get_task, list_tasks
from env.utils import make_processed_emails, recent_history, serialize_processed_emails


class EmailTriageEnv:
    """Deterministic OpenEnv environment for email triage workflows."""

    def __init__(self, task_level: TaskLevel = "easy"):
        self.task_level = task_level
        self.task_spec: TaskSpec | None = None
        self.grader: Grader | None = None
        self.processed_emails: list[ProcessedEmail] = []
        self.history: list[dict[str, Any]] = []
        self.current_idx = 0
        self.step_count = 0
        self.done = False
        self.final_score: float | None = None
        self.final_breakdown: dict[str, float] = {}
        self.reset(task_level=task_level)

    def reset(self, task_level: TaskLevel | None = None) -> Observation:
        """Reset the environment to one of the fixed tasks."""

        if task_level is not None:
            self.task_level = task_level

        self.task_spec = get_task(self.task_level)
        self.grader = Grader(self.task_spec)
        self.processed_emails = make_processed_emails(self.task_spec.emails)
        self.history = []
        self.current_idx = 0
        self.step_count = 0
        self.done = False
        self.final_score = None
        self.final_breakdown = {}
        return self._build_observation()

    def state(self) -> dict[str, Any]:
        """Return the full internal state for debugging and grading."""

        if self.task_spec is None:
            return {}

        return {
            "task_id": self.task_spec.task_id,
            "task_level": self.task_spec.task_level,
            "step_count": self.step_count,
            "max_steps": self.task_spec.max_steps,
            "ideal_steps": self.task_spec.ideal_steps,
            "current_index": self.current_idx,
            "done": self.done,
            "final_score": self.final_score,
            "final_breakdown": self.final_breakdown,
            "processed_emails": serialize_processed_emails(self.processed_emails),
            "history": self.history,
        }

    def step(self, action_dict: dict[str, Any] | Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        """Apply an action to the current email."""

        if self.task_spec is None or self.grader is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        if self.done:
            reward = build_reward(invalid_action=True, reason="Episode already completed.")
            return self._build_observation(), reward, True, self._final_info()

        self.step_count += 1

        try:
            action = action_dict if isinstance(action_dict, Action) else Action.model_validate(action_dict)
        except ValidationError as exc:
            reward = build_reward(
                invalid_action=True,
                extra_step=self._is_extra_step(),
                reason=f"Invalid action payload: {exc}",
            )
            self._append_history(email_id=self._current_email().email.id, action={"invalid_payload": action_dict}, reward=reward)
            self._check_done()
            return self._build_observation(), reward, self.done, self._final_info()

        current = self._current_email()
        truth = self.task_spec.expected_outcomes[current.email.id]

        if action.action_type not in self.task_spec.allowed_actions:
            reward = build_reward(
                invalid_action=True,
                extra_step=self._is_extra_step(),
                reason=f"Action '{action.action_type}' is not allowed for task '{self.task_spec.task_id}'.",
            )
            self._append_history(email_id=current.email.id, action=action.model_dump(), reward=reward)
            self._check_done()
            return self._build_observation(), reward, self.done, self._final_info()

        repeated_action = False
        classification_correct = False
        priority_correct = False
        action_correct = False
        folder_correct = False
        reason = f"Applied {action.action_type}."

        if action.action_type == "classify_email":
            if action.label is None:
                reward = build_reward(invalid_action=True, extra_step=self._is_extra_step(), reason="classify_email requires 'label'.")
                self._append_history(email_id=current.email.id, action=action.model_dump(), reward=reward)
                return self._build_observation(), reward, self.done, self._final_info()
            repeated_action = current.decision.label is not None
            current.decision.label = action.label
            classification_correct = action.label == truth.label
            reason = f"Classified email as {action.label}."

        elif action.action_type == "set_priority":
            if action.priority is None:
                reward = build_reward(invalid_action=True, extra_step=self._is_extra_step(), reason="set_priority requires 'priority'.")
                self._append_history(email_id=current.email.id, action=action.model_dump(), reward=reward)
                return self._build_observation(), reward, self.done, self._final_info()
            repeated_action = current.decision.priority is not None
            current.decision.priority = action.priority
            priority_correct = truth.priority is not None and action.priority == truth.priority
            reason = f"Set priority to {action.priority}."

        elif action.action_type == "take_action":
            if action.action_name is None:
                reward = build_reward(invalid_action=True, extra_step=self._is_extra_step(), reason="take_action requires 'action_name'.")
                self._append_history(email_id=current.email.id, action=action.model_dump(), reward=reward)
                return self._build_observation(), reward, self.done, self._final_info()
            repeated_action = current.decision.workflow_action is not None
            current.decision.workflow_action = action.action_name
            action_correct = truth.workflow_action is not None and action.action_name == truth.workflow_action
            reason = f"Chose workflow action {action.action_name}."

        elif action.action_type == "move_to_folder":
            if action.folder is None:
                reward = build_reward(invalid_action=True, extra_step=self._is_extra_step(), reason="move_to_folder requires 'folder'.")
                self._append_history(email_id=current.email.id, action=action.model_dump(), reward=reward)
                return self._build_observation(), reward, self.done, self._final_info()
            repeated_action = current.decision.folder is not None
            current.decision.folder = action.folder
            folder_correct = truth.folder is not None and action.folder == truth.folder
            reason = f"Moved email to {action.folder}."

        reward = build_reward(
            classification_correct=classification_correct,
            priority_correct=priority_correct,
            action_correct=action_correct,
            folder_correct=folder_correct,
            repeated_action=repeated_action,
            extra_step=self._is_extra_step(),
            reason=reason,
        )

        current.completed = self._email_is_complete(current)
        self._append_history(email_id=current.email.id, action=action.model_dump(), reward=reward)
        self._advance_to_next_email()
        self._check_done()

        if self.done and self.final_score is not None:
            reward.reason = f"{reward.reason} Episode completed with final score {self.final_score:.4f}."

        return self._build_observation(), reward, self.done, self._final_info()

    def _build_observation(self) -> Observation:
        assert self.task_spec is not None
        current_email = None if self.done else self.processed_emails[self.current_idx].email
        remaining_emails = [item.email for item in self.processed_emails if not item.completed and item.email.id != (current_email.id if current_email else None)]
        return Observation(
            task_id=self.task_spec.task_id,
            task_level=self.task_spec.task_level,
            instructions=self.task_spec.description,
            current_email=current_email,
            history_of_actions=recent_history(self.history),
            remaining_emails=remaining_emails,
            remaining_emails_count=len(remaining_emails),
            allowed_actions=self.task_spec.allowed_actions,
            step_count=self.step_count,
        )

    def _append_history(self, *, email_id: str, action: dict[str, Any], reward: Reward) -> None:
        self.history.append(
            {
                "step": self.step_count,
                "email_id": email_id,
                "action": action,
                "reward": reward.score,
                "reason": reward.reason,
                "breakdown": reward.breakdown,
            }
        )

    def _current_email(self) -> ProcessedEmail:
        return self.processed_emails[self.current_idx]

    def _email_is_complete(self, processed: ProcessedEmail) -> bool:
        if self.task_spec is None:
            return False
        if self.task_spec.task_level == "easy":
            return processed.decision.label is not None
        if self.task_spec.task_level == "medium":
            return processed.decision.label is not None and processed.decision.priority is not None
        return (
            processed.decision.label is not None
            and processed.decision.priority is not None
            and processed.decision.workflow_action is not None
            and processed.decision.folder is not None
        )

    def _advance_to_next_email(self) -> None:
        for index, processed in enumerate(self.processed_emails):
            if not processed.completed:
                self.current_idx = index
                return
        self.current_idx = len(self.processed_emails) - 1

    def _check_done(self) -> None:
        assert self.task_spec is not None and self.grader is not None
        if all(item.completed for item in self.processed_emails) or self.step_count >= self.task_spec.max_steps:
            self.done = True
            grade = self.grader.score(self.processed_emails, self.step_count)
            self.final_score = grade.score
            self.final_breakdown = grade.breakdown

    def _final_info(self) -> dict[str, Any]:
        if not self.done:
            return {}
        return {"final_score": self.final_score, "final_breakdown": self.final_breakdown}

    def _is_extra_step(self) -> bool:
        assert self.task_spec is not None
        return self.step_count > self.task_spec.ideal_steps


app = FastAPI(title="Email Triage OpenEnv")
api_env = EmailTriageEnv()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, list[str]]:
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset(payload: dict[str, str] | None = None) -> dict[str, Any]:
    task_level = (payload or {}).get("task_level", "easy")
    return api_env.reset(task_level=task_level).model_dump()


@app.post("/step")
def step(action: dict[str, Any]) -> dict[str, Any]:
    observation, reward, done, info = api_env.step(action)
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> dict[str, Any]:
    return api_env.state()
