from __future__ import annotations

import json
import os
import sys
import time
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
    """Run one task to completion and return metrics with mandatory formatted logs."""
    
    # [START] tag in exact format requested: [START] task=NAME
    print(f"[START] task={task_level}", flush=True)
    
    env = EmailTriageEnv(task_level=task_level)
    observation = env.reset(task_level=task_level)
    done = False
    cumulative_reward = 0.0
    step_num = 1

    while not done:
        action = infer_action(client, model_name, observation.model_dump())
        observation, reward, done, _info = env.step(action)
        cumulative_reward += reward.score
        
        # [STEP] tag in exact format requested: [STEP] step=N reward=VAL
        print(f"[STEP] step={step_num} reward={reward.score} cumulative_reward={round(cumulative_reward, 4)} done={done}", flush=True)
        
        step_num += 1

    state = env.state()
    final_score = state["final_score"]
    steps_taken = state["step_count"]
    
    # [END] tag in exact format requested: [END] task=NAME score=VAL steps=N
    print(f"[END] task={task_level} score={final_score} steps={steps_taken}", flush=True)
    
    return {
        "task": task_level,
        "steps": steps_taken,
        "cumulative_reward": round(cumulative_reward, 4),
        "final_score": final_score,
        "final_breakdown": state["final_breakdown"],
    }


def main() -> None:
    """Run all tasks and print reproducible scores."""
    
    # Verify environment
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: Missing OPENAI_API_KEY environment variable.", flush=True)
        return

    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    client = create_client()
    
    start_time = time.time()
    results = [run_task(task_level, client, model_name) for task_level in list_tasks()]
    end_time = time.time()
    
    average_score = round(sum(item["final_score"] for item in results) / len(results), 4)
    summary = {
        "model_name": model_name,
        "total_runtime_sec": round(end_time - start_time, 2),
        "average_score": average_score,
        "results": results
    }
    
    # Final overall summary for the human judges
    print("\n--- FINAL BENCHMARK SUMMARY ---", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    # Ensure stdout is unbuffered at the system level
    main()
