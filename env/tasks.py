from __future__ import annotations

from env.models import Email, EmailExpectation, TaskLevel, TaskSpec


def list_tasks() -> list[str]:
    return ["easy", "medium", "hard"]


def get_task(task_level: TaskLevel) -> TaskSpec:
    task_map = {"easy": _easy_task(), "medium": _medium_task(), "hard": _hard_task()}
    if task_level not in task_map:
        raise ValueError(f"Unknown task: {task_level}")
    return task_map[task_level]


def _easy_task() -> TaskSpec:
    emails = [
        Email(
            id="e1",
            subject="Win a free PyTorch certification today!!",
            body="Click here to claim your free PyTorch voucher. Don't miss out on this one time offer...",
            sender="offers@spam-torch.biz",
            timestamp="2026-03-01T08:10:00Z",
            metadata={"contains_attachment": True, "campaign": True},
        ),
        Email(
            id="e2",
            subject="Help: Discord server bot is down before demo",
            body="Hey team, the Scaler bot just went offline and we have a presentation in 20 minutes. Can someone check the logs?",
            sender="mentor@scaler.com",
            timestamp="2026-03-01T08:18:00Z",
            metadata={"customer_tier": "enterprise", "business_impact": "high"},
        ),
        Email(
            id="e3",
            subject="Weekly Github commits digest",
            body="Here is your automated report for repo activity this week.",
            sender="reports@github.com",
            timestamp="2026-03-01T08:30:00Z",
            metadata={"automated": True},
        ),
        Email(
            id="e4",
            subject="Urgent: Verify your Meta OpenEnv account details",
            body="Please reply with your password to verify your hackathon eligibility immediately.",
            sender="security@fishy-meta-login.net",
            timestamp="2026-03-01T08:35:00Z",
            metadata={"spoof_warning": True},
        ),
    ]
    expected = {
        "e1": EmailExpectation(label="spam"),
        "e2": EmailExpectation(label="important"),
        "e3": EmailExpectation(label="normal"),
        "e4": EmailExpectation(label="spam"),
    }
    return TaskSpec(
        task_id="easy",
        task_level="easy",
        description="Just tag the emails as spam, important, or normal.",
        allowed_actions=["classify_email"],
        emails=emails,
        expected_outcomes=expected,
        max_steps=10,
        ideal_steps=4,
    )


def _medium_task() -> TaskSpec:
    emails = [
        Email(
            id="m1",
            subject="GPU quota exceeded on our EC2 instance",
            body="We just hit our limit for AWS GPUs. We need to shut down some idle training jobs right now to avoid a massive bill.",
            sender="devops@team.com",
            timestamp="2026-03-02T09:00:00Z",
            metadata={"sentiment": "negative", "business_impact": "high"},
        ),
        Email(
            id="m2",
            subject="Draft review: Hackathon submission video",
            body="Take a look at the video edit for our OpenEnv submission when you have a sec.",
            sender="teammate@scaler.com",
            timestamp="2026-03-02T09:05:00Z",
            metadata={"internal": True},
        ),
        Email(
            id="m3",
            subject="API key leaked on public repl",
            body="Warning: We detected an active Hugging Face token in a recent public commit. Please revoke it ASAP.",
            sender="alerts@gitguardian.com",
            timestamp="2026-03-02T09:08:00Z",
            metadata={"security_signal": True},
        ),
        Email(
            id="m4",
            subject="Boost your LLM performance by 500% with this one weird trick",
            body="Buy our ebook to find out how to hack standard benchmarks.",
            sender="marketing@scam-ai.co",
            timestamp="2026-03-02T09:10:00Z",
            metadata={"campaign": True},
        ),
    ]
    expected = {
        "m1": EmailExpectation(label="important", priority="high"),
        "m2": EmailExpectation(label="normal", priority="low"),
        "m3": EmailExpectation(label="important", priority="high"),
        "m4": EmailExpectation(label="spam", priority="low"),
    }
    return TaskSpec(
        task_id="medium",
        task_level="medium",
        description="Tag emails and assign priority (low, medium, high).",
        allowed_actions=["classify_email", "set_priority"],
        emails=emails,
        expected_outcomes=expected,
        max_steps=16,
        ideal_steps=8,
    )


def _hard_task() -> TaskSpec:
    emails = [
        Email(
            id="h1",
            subject="Re: PyTorch training loop crashing on epoch 2",
            body="I tried changing the batch size but it still segfaults. The deadline is tomorrow to submit this to Scaler, someone please look into the CUDA error logs.",
            sender="teammate1@school.edu",
            timestamp="2026-03-03T07:45:00Z",
            metadata={"thread_length": 3, "business_impact": "high"},
        ),
        Email(
            id="h2",
            subject="ACTION REQUIRED: Unpaid invoice #77123",
            body="Your server hosting will be deleted in 1 hour if you don't click this link to pay.",
            sender="billing@shady-hoster.co",
            timestamp="2026-03-03T07:50:00Z",
            metadata={"spoof_warning": True},
        ),
        Email(
            id="h3",
            subject="What color should our team logo be?",
            body="I was thinking maybe purple or green for our OpenEnv submission. Any thoughts?",
            sender="teammate2@school.edu",
            timestamp="2026-03-03T08:00:00Z",
            metadata={"business_impact": "low"},
        ),
        Email(
            id="h4",
            subject="Question about reward logic scaling",
            body="Hey, I was reading the docs on our dense rewards. Can we sync up tomorrow to discuss how to normalize it?",
            sender="teammate1@school.edu",
            timestamp="2026-03-03T08:02:00Z",
            metadata={"customer_tier": "internal"},
        ),
        Email(
            id="h5",
            subject="Docker build failing on main branch",
            body="It looks like the base image we used update their python version and our pip is breaking. Blocking all deployments right now.",
            sender="ci-bot@github.com",
            timestamp="2026-03-03T08:05:00Z",
            metadata={"thread_length": 2, "business_impact": "medium"},
        ),
    ]
    expected = {
        "h1": EmailExpectation(label="important", priority="high", workflow_action="escalate", folder="support"),
        "h2": EmailExpectation(label="spam", priority="low", workflow_action="archive", folder="spam"),
        "h3": EmailExpectation(label="normal", priority="low", workflow_action="schedule", folder="calendar"),
        "h4": EmailExpectation(label="important", priority="medium", workflow_action="respond", folder="priority"),
        "h5": EmailExpectation(label="important", priority="high", workflow_action="respond", folder="support"),
    }
    return TaskSpec(
        task_id="hard",
        task_level="hard",
        description="Full workflow: tag, prioritize, pick workflow action, and folder routing.",
        allowed_actions=["classify_email", "set_priority", "take_action", "move_to_folder"],
        emails=emails,
        expected_outcomes=expected,
        max_steps=25,
        ideal_steps=20,
    )
