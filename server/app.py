import gradio as gr
import os
import sys
import os

# Add parent directory to sys.path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline import get_client, run_task
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from email_env import EmailEnv
from schema import Action as PydanticAction
import uvicorn

load_dotenv()

# --- REST API Implementation for OpenEnv Compliance ---
app = FastAPI()
# Global environment instance for API calls
global_env = EmailEnv(task_id="easy")

@app.post("/reset")
async def reset_env(request: Request):
    # Some validators might send a seed in the JSON body
    seed = None
    try:
        body = await request.json()
        seed = body.get("seed")
    except:
        pass
    obs = global_env.reset(seed=seed)
    return obs.model_dump()

@app.post("/step")
async def step_env(action: PydanticAction):
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.post("/state")
async def get_state():
    return global_env.state().model_dump()

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Gradio UI Implementation ---
def run_evaluation_generator(num_trials):
    try:
        client, model = get_client()
    except ValueError as e:
        yield f"### ❌ Error\n{e}. Please ensure your API keys are set in the Settings > Secrets on Hugging Face."
        return

    num_trials = int(num_trials)
    results = {}
    summary = "### 🚀 Starting Benchmark...\n"
    yield summary

    for task_id in ["easy", "medium", "hard"]:
        scores = []
        summary += f"\n**Task: {task_id.capitalize()}**\n"
        yield summary
        
        for trial in range(num_trials):
            # Update UI that a trial is starting
            current_status = summary + f"  - ⏳ Trial {trial+1}/{num_trials} in progress..."
            yield current_status
            
            # Run the actual task
            score = run_task(client, model, task_id, seed=42 + trial)
            scores.append(score)
            
            # Update summary with the result of this trial
            summary += f"  - ✅ Trial {trial+1}/{num_trials} complete (Score: {score})\n"
            yield summary
        
        avg_score = sum(scores) / len(scores)
        results[task_id] = {"avg_score": avg_score, "all_scores": scores}
    
    # Final formatted summary
    final_summary = "## 🏆 Final Baseline Results Summary\n\n"
    for task_id, data in results.items():
        final_summary += f"### {task_id.capitalize()} Task\n"
        final_summary += f"- **Success Rate**: {data['avg_score'] * 100:.1f}%\n"
        final_summary += f"- **Individual Scores**: `{data['all_scores']}`\n\n"
    
    final_summary += "---\n*Benchmark complete. You can run it again with a different number of trials.*"
    yield final_summary

with gr.Blocks(title="OpenEnv: Email Triage Benchmark", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📧 OpenEnv: Email Triage & Calendar Management")
    gr.Markdown("""
    This environment simulates real-world email triage and calendar management tasks. 
    Click **Run Benchmark** to see the AI agent (Llama-3 via Groq) handle the inbox in real-time.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            trials_input = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Number of Trials per Task")
            run_btn = gr.Button("▶️ Run Benchmark", variant="primary")
        
        with gr.Column(scale=2):
            output_markdown = gr.Markdown("### 🕒 Status\nClick 'Run Benchmark' to start...")

    gr.Markdown("---")
    with gr.Accordion("📖 View Tasks Description", open=False):
        gr.Markdown("""
        - **Easy (Newsletter Cleanup)**: The agent must identify and archive 5 newsletter emails while ignoring personal ones.
        - **Medium (Meeting Coordination)**: The agent must reply to a meeting request and correctly create a calendar event with details.
        - **Hard (Support Escalation)**: The agent must identify an urgent production issue, mark it urgent, forward it to engineering, and reply to the customer.
        """)

    run_btn.click(
        fn=run_evaluation_generator,
        inputs=[trials_input],
        outputs=[output_markdown]
    )

def main():
    # Hugging Face Spaces uses port 7860 by default
    global app
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
