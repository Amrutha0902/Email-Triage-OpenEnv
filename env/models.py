from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


EmailLabel = Literal["spam", "important", "normal"]
PriorityLevel = Literal["low", "medium", "high"]
WorkflowAction = Literal["archive", "respond", "escalate", "schedule"]
FolderName = Literal["spam", "inbox", "priority", "support", "calendar", "archive"]
TaskLevel = Literal["easy", "medium", "hard"]


class Email(BaseModel):
    """Single inbox email presented to the agent."""

    id: str = Field(..., description="Unique identifier for the email.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Email body content.")
    sender: str = Field(..., description="Sender address.")
    timestamp: str = Field(..., description="ISO-8601 delivery timestamp.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Structured metadata for triage reasoning.")


class EmailDecision(BaseModel):
    """Mutable per-email decisions recorded by the environment."""

    label: Optional[EmailLabel] = None
    priority: Optional[PriorityLevel] = None
    workflow_action: Optional[WorkflowAction] = None
    folder: Optional[FolderName] = None


class ProcessedEmail(BaseModel):
    """Email plus the current agent decision state."""

    email: Email
    decision: EmailDecision = Field(default_factory=EmailDecision)
    completed: bool = False


class Observation(BaseModel):
    """OpenEnv observation returned each step."""

    task_id: str = Field(..., description="Current task identifier.")
    task_level: TaskLevel = Field(..., description="Difficulty of the active task.")
    instructions: str = Field(..., description="Natural-language objective for the agent.")
    current_email: Optional[Email] = Field(None, description="Email currently being triaged.")
    history_of_actions: list[dict[str, Any]] = Field(default_factory=list, description="Recent action history.")
    remaining_emails: list[Email] = Field(default_factory=list, description="Unfinished emails remaining in the queue.")
    remaining_emails_count: int = Field(..., description="Count of remaining unfinished emails.")
    allowed_actions: list[str] = Field(default_factory=list, description="Action types allowed for the active task.")
    step_count: int = Field(..., description="How many steps have been taken in the current episode.")


class Action(BaseModel):
    """Structured action accepted by the environment."""

    action_type: Literal["classify_email", "set_priority", "take_action", "move_to_folder"] = Field(
        ..., description="Action type to apply to the current email."
    )
    label: Optional[EmailLabel] = Field(None, description="Label used with classify_email.")
    priority: Optional[PriorityLevel] = Field(None, description="Priority used with set_priority.")
    action_name: Optional[WorkflowAction] = Field(None, description="Workflow action used with take_action.")
    folder: Optional[FolderName] = Field(None, description="Folder used with move_to_folder.")


class Reward(BaseModel):
    """Dense reward object returned by each step."""

    score: float = Field(..., description="Step reward in the range [-1.0, 1.0].")
    reason: str = Field(..., description="Human-readable explanation of the reward.")
    breakdown: dict[str, float] = Field(default_factory=dict, description="Reward component breakdown.")


class EmailExpectation(BaseModel):
    """Ground-truth outcome used by the deterministic grader."""

    label: EmailLabel
    priority: Optional[PriorityLevel] = None
    workflow_action: Optional[WorkflowAction] = None
    folder: Optional[FolderName] = None


class TaskSpec(BaseModel):
    """Fixed task definition for one difficulty level."""

    task_id: str
    task_level: TaskLevel
    description: str
    allowed_actions: list[str]
    emails: list[Email]
    expected_outcomes: dict[str, EmailExpectation]
    max_steps: int
    ideal_steps: int
