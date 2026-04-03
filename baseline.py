import os
import json
from typing import List
from openai import OpenAI
from email_env import EmailEnv
from schema import Action, ActionType
from dotenv import load_dotenv

load_dotenv()

def get_client():
    # Try Groq first as an alternative
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        print("Using Groq API...")
        return OpenAI(
            api_key=groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        ), "llama3-8b-8192"

    # Fallback to OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        print("Using OpenAI API...")
        return OpenAI(api_key=openai_api_key), "gpt-4o"
    
    raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY is set. Please set one in your .env file.")

def get_action_from_llm(client: OpenAI, model: str, observation_dict: dict, task_id: str) -> Action:
    """Uses LLM to decide on the next action based on the observation."""
    
    base_prompt = f"""
    You are an AI assistant managing an inbox. Your goal is to process emails and manage your calendar.
    You MUST respond ONLY with a JSON object representing the action.
    The action MUST adhere to the following Pydantic schema:
    {{
        "action_type": "one of: archive, reply, forward, mark_urgent, create_calendar_event, wait",
        "email_id": "optional: ID of the email to act on",
        "content": "optional: for reply or forward",
        "recipient": "optional: for forward",
        "event_details": "optional: for create_calendar_event (e.g., {{"title": "Meeting", "time": "2024-03-22 10:00"}})"
    }}
    
    If no specific email_id is provided for an action, assume it applies to the most relevant email based on the task.
    If you cannot determine a valid action, use {{"action_type": "wait"}}.
    
    # Token optimization: Remove redundant info from observation
    optimized_obs = {
        "inbox": [{"id": e["id"], "sender": e["sender"], "subject": e["subject"]} for e in observation_dict["inbox"]],
        "unread_count": observation_dict["unread_count"],
        "calendar_events": observation_dict["calendar_events"]
    }
    
    Current Observation (Summarized):
    {json.dumps(optimized_obs, indent=2)}
    
    ---
    """

    task_specific_instructions = ""
    if task_id == "easy":
        task_specific_instructions = """
        Your objective is to **archive all newsletter emails**. Ignore all other emails for this task.
        Newsletters are typically from senders like 'newsletter@weekly.com' and have subjects like 'Weekly Update'.
        Example: {{"action_type": "archive", "email_id": "news_0"}}
        """
    elif task_id == "medium":
        task_specific_instructions = """
        Your objective is to **reply to the boss's meeting request and create a calendar event** for it.
        The boss's email will contain details about the meeting time and subject.
        Example:
        {{"action_type": "reply", "email_id": "boss_1", "content": "Confirmed, I will add it to the calendar."}}
        {{"action_type": "create_calendar_event", "event_details": {{"title": "Quarterly Review", "time": "Friday at 10 AM"}} }}
        """
    elif task_id == "hard":
        task_specific_instructions = """
        Your objective is to handle an **urgent production down escalation**. You MUST perform three actions:
        1. **Mark the urgent customer email as urgent.**
        2. **Forward the urgent customer email to 'engineering@company.com'.**
        3. **Reply to the customer, acknowledging the issue and stating that the team is investigating.**
        Prioritize the urgent customer email.
        Example:
        {{"action_type": "mark_urgent", "email_id": "cust_1"}}
        {{"action_type": "forward", "email_id": "cust_1", "recipient": "engineering@company.com", "content": "Urgent production issue, please investigate."}}
        {{"action_type": "reply", "email_id": "cust_1", "content": "We are investigating the production issue and will provide an update shortly."}}
        """
    
    prompt = base_prompt + task_specific_instructions
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise office assistant. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_llm_response = response.choices[0].message.content
        print(f"LLM Raw Response: {raw_llm_response}") # Debug print
        action_data = json.loads(raw_llm_response)
        return Action(**action_data)
    except Exception as e:
        print(f"Error calling LLM ({model}): {e}")
        return Action(action_type=ActionType.WAIT)

def run_task(client: OpenAI, model: str, task_id: str, seed: int = 42):
    print(f"\n--- Running Task: {task_id} (Seed: {seed}) ---")
    env = EmailEnv(task_id=task_id)
    obs = env.reset(seed=seed)
    done = False
    step = 0
    total_reward = 0.0
    
    while not done and step < 10:
        print(f"\n--- Step {step+1} ---")
        print(f"Observation sent to LLM: {json.dumps(obs.model_dump(), indent=2)}") # Debug print
        
        # 1. Get action from LLM
        action = get_action_from_llm(client, model, obs.model_dump(), task_id)
        print(f"Agent chose {action.action_type} for {action.email_id if action.email_id else 'N/A'}")
        
        # 2. Step the environment
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        step += 1
        
        print(f"Reward: {reward.value}, Done: {done}, Info: {info}") # Debug print
        
        if done:
            print(f"Task finished. Final Score: {info['score']}")
            return info['score']
    
    final_score = env.grade()
    print(f"Task timed out. Final Score: {final_score}")
    return final_score

def run_benchmark(client, model, num_trials: int = 3):
    print(f"\n--- Starting Full Benchmark ({num_trials} trials per task) ---")
    results = {}
    
    for task_id in ["easy", "medium", "hard"]:
        scores = []
        for trial in range(num_trials):
            # Each trial gets a predictable seed based on its trial number
            score = run_task(client, model, task_id, seed=42 + trial)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        results[task_id] = {
            "avg_score": avg_score,
            "all_scores": scores
        }
    
    return results

def main():
    try:
        client, model = get_client()
    except ValueError as e:
        print(f"Error: {e}")
        return

    benchmark_results = run_benchmark(client, model, num_trials=3)
    
    print("\n--- Reproducible Baseline Results Summary ---")
    for task_id, data in benchmark_results.items():
        print(f"Task {task_id}: {data['avg_score'] * 100:.1f}% average success (Scores: {data['all_scores']})")

if __name__ == "__main__":
    main()
