from smart_home import SmartHomeEnv
import random

class RandomAgent:
    """A simple agent that takes random actions."""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

def main():
    # 1. Initialize environment and agent
    env = SmartHomeEnv()
    agent = RandomAgent(env.action_space)

    print("--- Starting Smart Home Energy Management Simulation ---")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")

    # 2. Reset environment
    observation = env.reset(seed=42)
    total_reward = 0.0
    done = False
    step_count = 0

    print(f"\nInitial State: {observation}")
    print(f"{'Step':<5} | {'Action':<10} | {'Reward':<10} | {'Indoor Temp':<12} | {'Battery %':<10} | {'Price':<10}")
    print("-" * 75)

    # 3. Simulation Loop
    while not done and step_count < 24: # Run for 24 hours (1 day)
        # Agent decides on action
        action = agent.act(observation)
        
        # Environment takes a step
        next_observation, reward, done, info = env.step(action)
        
        # Track progress
        total_reward += reward
        step_count += 1
        
        # Log every hour
        print(f"{step_count:<5} | {action:<10} | {reward:10.2f} | {info['indoor_temp']:12.2f} | {info['battery_level']:10.1f} | {next_observation[4]:10.2f}")
        
        # Update observation
        observation = next_observation

    print("-" * 75)
    print(f"Simulation Finished after {step_count} steps.")
    print(f"Total Reward: {total_reward:.2f}")
    
    # 4. State API test
    print(f"\nCurrent Environment State (state() API): {env.state()}")

if __name__ == "__main__":
    main()
