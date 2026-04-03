import gradio as gr
import os
from baseline import get_client, run_task
from dotenv import load_dotenv

load_dotenv()

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

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
