from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from env.environment import EmailTriageEnv
from env.models import Action
from env.tasks import list_tasks


SYSTEM_PROMPT = """
You are controlling an email triage environment.
Return exactly one JSON object using this schema:
{
  "action_type": "classify_email" | "set_priority" | "take_action" | "move_to_folder",
  "label": "spam" | "important" | "normal" | null,
  "priority": "low" | "medium" | "high" | null,
  "action_name": "archive" | "respond" | "escalate" | "schedule" | null,
  "folder": "spam" | "inbox" | "priority" | "support" | "calendar" | "archive" | null
}
Only populate the field relevant to the chosen action_type.
Do not include explanations.
""".strip()


def create_client() -> OpenAI:
    """Create an OpenAI client from environment variables."""

    api_key = os.environ["OPENAI_API_KEY"]
    base_url = os.environ.get("API_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def build_prompt(observation: dict[str, Any]) -> str:
    """Build a stable prompt for deterministic inference."""

    return (
        "You are solving the next step of an email triage task.\n"
        "Observation JSON:\n"
        f"{json.dumps(observation, sort_keys=True, indent=2)}\n"
        "Return JSON only."
    )


def infer_action(client: OpenAI, model_name: str, observation: dict[str, Any]) -> Action:
    """Query the model for one structured action."""

    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(observation)},
        ],
        response_format={ "type": "json_object" }
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned empty response.")
    return Action.model_validate_json(content.strip())


def run_task(task_level: str, client: OpenAI, model_name: str) -> dict[str, Any]:
    """Run one task to completion and return deterministic metrics."""

    env = EmailTriageEnv(task_level=task_level)
    observation = env.reset(task_level=task_level)
    done = False
    cumulative_reward = 0.0

    while not done:
        action = infer_action(client, model_name, observation.model_dump())
        observation, reward, done, _info = env.step(action)
        cumulative_reward += reward.score

    state = env.state()
    return {
        "task": task_level,
        "steps": state["step_count"],
        "cumulative_reward": round(cumulative_reward, 4),
        "final_score": state["final_score"],
        "final_breakdown": state["final_breakdown"],
    }


def main() -> None:
    """Run all tasks and print reproducible scores."""

    _ = os.environ.get("HF_TOKEN", "")
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
    client = create_client()
    results = [run_task(task_level, client, model_name) for task_level in list_tasks()]
    average_score = round(sum(item["final_score"] for item in results) / len(results), 4)
    print(json.dumps({"model_name": model_name, "results": results, "average_score": average_score}, indent=2))


if __name__ == "__main__":
    main()
