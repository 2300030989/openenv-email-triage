import os
import json
import time
from typing import List
from openai import OpenAI
from email_env import EmailEnv
from schema import Action, ActionType
from dotenv import load_dotenv

load_dotenv()

# Required environment variables for submission
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
# Try to get API key from GROQ or OPENAI depending on the provider
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def get_action_from_llm(observation_dict: dict, task_id: str) -> Action:
    prompt = f"""
    Objective: {task_id}
    Current Observation: {json.dumps(observation_dict)}
    Respond with JSON: {{"action_type": "...", "email_id": "...", ...}}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        action_data = json.loads(response.choices[0].message.content)
        return Action(**action_data)
    except Exception:
        return Action(action_type=ActionType.WAIT)

def run_inference():
    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        env = EmailEnv(task_id=task_id)
        obs = env.reset(seed=42)
        
        # [START] mandatory log
        print(f"[START] task_id={task_id}")
        
        done = False
        step = 0
        while not done and step < 10:
            step += 1
            action = get_action_from_llm(obs.model_dump(), task_id)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            # [STEP] mandatory log
            # Format: [STEP] step=N, action=ACTION, reward=R, done=D
            print(f"[STEP] step={step}, action={action.action_type}, reward={reward.value}, done={done}")
            
            obs = next_obs
            
        # [END] mandatory log
        # Format: [END] task_id=T, score=S
        print(f"[END] task_id={task_id}, score={env.grade()}")

if __name__ == "__main__":
    run_inference()
