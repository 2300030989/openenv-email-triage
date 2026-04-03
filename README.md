---
title: OpenEnv Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Email Triage & Calendar Management Environment (OpenEnv)

## Environment Description & Motivation
This environment simulates a professional email management task—something millions of people do daily. Unlike games or toy environments, this task requires understanding natural language, identifying urgency, making decisions about archiving, and coordinating with a calendar. 

The goal is to provide a realistic benchmark for AI agents to demonstrate their ability to handle asynchronous communication and scheduling tasks autonomously.

## Action & Observation Space Definitions

### Action Space (Discrete)
- `archive`: Removes an email from the inbox.
- `reply`: Sends a response to the sender (requires `content`).
- `forward`: Forwards an email to a new recipient (requires `recipient` and `content`).
- `mark_urgent`: Flags an email as high priority.
- `create_calendar_event`: Adds an event to the calendar (requires `event_details`).
- `wait`: Do nothing for one step.

### Observation Space (Box/Typed)
- `inbox`: A list of `Email` objects (sender, subject, body, timestamp, flags).
- `unread_count`: Integer representing how many emails are still unread.
- `calendar_events`: List of scheduled items.
- `last_action_status`: Feedback on the previous action taken.

## Task Descriptions

| Task ID | Name | Difficulty | Expected Behavior |
|---------|------|------------|-------------------|
| `easy` | Newsletter Cleanup | Easy | Agent must archive 5 specific newsletter emails. |
| `medium` | Meeting Coordination | Medium | Agent must reply to a meeting request and create a calendar event. |
| `hard` | Support Escalation | Hard | Agent must mark a customer email as urgent, forward it to engineering, and reply to the customer. |

## Setup & Usage

### Prerequisites
- Python 3.12+
- OpenAI API Key (for baseline inference)

### Installation
```bash
pip install -r requirements.txt
```

### Running the Environment
You can run the baseline script to see how a model performs on these tasks.

1. Create a `.env` file and add your key:
```bash
OPENAI_API_KEY="your_api_key_here"
```
2. Or set it in your shell:
```bash
# PowerShell
$env:OPENAI_API_KEY="your_api_key"

# Bash
export OPENAI_API_KEY="your_api_key"
```
3. Run the script:
```bash
python baseline.py
```

### Containerization
To build and run locally with Docker:
```bash
docker build -t openenv-email .
docker run -e OPENAI_API_KEY="your_api_key" openenv-email
```

## Baseline Scores (Reproducible)
The baseline uses `llama-3.3-70b-versatile` (via Groq) or `gpt-4o` (via OpenAI). Results are averaged across multiple trials with fixed seeds for reproducibility.

| Task | Avg. Score | Detailed Scores |
|------|------------|-----------------|
| Easy | 40% | [0.4, 0.4, 0.4] |
| Medium | 100% | [1.0, 1.0, 1.0] |
| Hard | 34% | [0.34, 0.34, 0.34] |

## Deployment to Hugging Face Spaces
This environment is designed to run as a containerized Hugging Face Space.

1.  **Create a New Space**: Select "Docker" as the SDK.
2.  **Upload Files**: Push all files in this repository (excluding `.env`).
3.  **Set Secrets**: In your Space's Settings, add `GROQ_API_KEY` or `OPENAI_API_KEY` as a Secret.
4.  **Automatic Build**: Hugging Face will automatically build the `Dockerfile` and launch the Gradio UI on port 7860.

## OpenEnv Compliance
This environment fully implements the OpenEnv specification:
- **Typed Models**: Uses Pydantic for `Observation`, `Action`, and `Reward`.
- **Core API**: Implements `reset()`, `step()`, and `state()`.
- **Metadata**: Includes `openenv.yaml` with environment metadata.
