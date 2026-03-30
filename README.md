---
title: Email Triage Env
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags: [openenv]
---

# Email Triage System (Meta x Scaler Hackathon)

Hey! Welcome to my submission for Round 1 of the Meta PyTorch OpenEnv Hackathon. 

Instead of doing a basic mini-game, I decided to tackle a real problem: **Email Triage**. In real organizations, developers and operations teams get flooded with emails ranging from critical server crashes down to standard junk. Agents should be able to look at the metadata, context, and subject to figure out exactly what to do.

## OpenEnv Setup
I strictly followed the gymnasium-style OpenEnv specs provided by the hackathon. We have `reset()`, `step()`, and `state()` endpoints hooked up through a fast Dockerized API. Pydantic handles all the structured typing so it doesn't break during grading.

## How the Agent interacts:
When an LLM agent starts, it's presented with an email block containing the subject, body, sender, timestamps, and some metadata flags. 

### Action Space options
* `classify_email(label)`: Is it spam, normal, or important?
* `set_priority(level)`: Needs low, medium, or high priority based on urgency vs importance tradeoffs.
* `take_action(action_type)`: Pick from archive, respond, escalate, or schedule.
* `move_to_folder(folder)`: Place it in the correct inbox or spam bin.

## Tasks included
* **Easy**: Just binary classification.
* **Medium**: Learn how to classify and also rank priority.
* **Hard**: Full routing flow with ambiguous LLM logic (distinguishing between true emergencies vs phishing campaigns).

## How Scoring actually works (Rewards)
I wanted the reward function to be super dense. If an agent at least figures out an email is "spam" but routes it to the wrong folder, it should still get partial credit instead of a binary zero!

* Correct classification: +0.30
* Correct priority: +0.30
* Correct action: +0.25
* Correct folder: +0.15
* Doing something weird/invalid: -0.10
* Efficiency: Points are slightly docked if the agent wastes multiple steps trying to loop on the same email.

## Setup Instructions
You can spin this whole thing up on Docker (which is what Hugging Face is currently doing in the background). 

For local testing:
```bash
pip install -r requirements.txt
python -m uvicorn env.environment:app --host 0.0.0.0 --port 7860
```

To run inference:
```bash
export OPENAI_API_KEY="your_key"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"

python inference.py
```
