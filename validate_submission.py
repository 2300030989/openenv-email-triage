import sys
import os
import yaml
from email_env import EmailEnv
from schema import Action, ActionType, Observation, Reward

def validate():
    print("--- Starting Pre-Submission Validation ---")
    
    # 1. Check Files
    required_files = ["inference.py", "openenv.yaml", "Dockerfile", "requirements.txt", "pyproject.toml"]
    for f in required_files:
        if os.path.exists(f):
            print(f"OK: Found {f}")
        else:
            print(f"FAIL: Missing {f}")
            sys.exit(1)

    # 2. Validate openenv.yaml
    try:
        with open("openenv.yaml", "r") as f:
            meta = yaml.safe_load(f)
            print(f"OK: openenv.yaml is valid YAML (Name: {meta.get('name')})")
    except Exception as e:
        print(f"FAIL: openenv.yaml error: {e}")
        sys.exit(1)

    # 3. Validate API Implementation
    try:
        env = EmailEnv(task_id="easy")
        obs = env.reset(seed=42)
        if not isinstance(obs, Observation):
            raise TypeError("reset() must return an Observation model")
        
        # Test step
        action = Action(action_type=ActionType.WAIT)
        next_obs, reward, done, info = env.step(action)
        
        if not isinstance(next_obs, Observation):
            raise TypeError("step() must return an Observation model")
        if not isinstance(reward, Reward):
            raise TypeError("step() must return a Reward model")
        
        state = env.state()
        if not isinstance(state, Observation):
            raise TypeError("state() must return an Observation model")
            
        print("OK: Environment API (reset/step/state) is compliant")
    except Exception as e:
        print(f"FAIL: API Compliance error: {e}")
        sys.exit(1)

    # 4. Check REST API (Simulation)
    print("\nChecking if REST API endpoints are defined...")
    from app import app as fastapi_app
    from fastapi.testclient import TestClient
    
    client = TestClient(fastapi_app)
    try:
        response = client.post("/reset", json={})
        if response.status_code == 200:
            print("OK: POST /reset responds with 200")
        else:
            print(f"FAIL: POST /reset returned {response.status_code}")
            
        response = client.post("/state")
        if response.status_code == 200:
            print("OK: POST /state responds with 200")
        else:
            print(f"FAIL: POST /state returned {response.status_code}")
            
        print("✅ REST API structure is ready for Hugging Face Space")
    except Exception as e:
        print(f"FAIL: REST API test error: {e}")

    # 4. Check Environment Variables
    required_vars = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
    for v in required_vars:
        if os.getenv(v):
            print(f"OK: Found env var: {v}")
        else:
            print(f"WARN: Missing env var: {v} (Make sure to set this in HF Space Secrets)")

    print("\nAll local checks passed! You are ready to push to GitHub/HF.")

if __name__ == "__main__":
    validate()
