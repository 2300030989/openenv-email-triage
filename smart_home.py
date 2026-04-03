import numpy as np
from typing import Tuple, Dict, Any
from base import OpenEnv
from spaces import Discrete, Box

class SmartHomeEnv(OpenEnv):
    """
    A real-world Smart Home Energy Management environment.
    The agent manages indoor temperature and battery storage to minimize costs
    while maintaining comfort.
    """

    def __init__(self):
        # State: [Indoor Temp, Outdoor Temp, Battery %, Hour, Electricity Price]
        # Temp in Celsius, Battery in %, Price in $/kWh
        self._observation_space = Box(
            low=np.array([10.0, -10.0, 0.0, 0.0, 0.05]),
            high=np.array([35.0, 45.0, 100.0, 23.0, 0.50]),
            shape=(5,)
        )
        
        # Actions:
        # 0: Idle
        # 1: AC On (Cooling)
        # 2: Heater On (Heating)
        # 3: Charge Battery
        # 4: Discharge Battery
        self._action_space = Discrete(5)

        self.target_temp = 22.0
        self.reset()

    def reset(self, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        
        # Initial state
        self.indoor_temp = 22.0 + np.random.uniform(-2, 2)
        self.outdoor_temp = 25.0 + np.random.uniform(-5, 5)
        self.battery_level = 50.0 # Start at 50%
        self.hour = 0
        self.price = self._get_electricity_price(self.hour)
        
        self.current_step = 0
        self.max_steps = 24 * 7 # One week simulation (hourly steps)
        
        return self.state()

    def _get_electricity_price(self, hour: int) -> float:
        """Simulate time-of-use electricity pricing."""
        if 8 <= hour <= 20: # Peak hours
            return 0.35
        else: # Off-peak
            return 0.15

    def state(self) -> np.ndarray:
        return np.array([
            self.indoor_temp,
            self.outdoor_temp,
            self.battery_level,
            float(self.hour),
            self.price
        ], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # 1. Update Environment Physics
        # Natural temperature drift towards outdoor temperature
        drift = (self.outdoor_temp - self.indoor_temp) * 0.05
        self.indoor_temp += drift

        # 2. Process Action
        cost = 0.0
        energy_used = 0.0
        
        if action == 1: # AC On
            self.indoor_temp -= 1.5
            energy_used = 2.0 # kWh
        elif action == 2: # Heater On
            self.indoor_temp += 1.5
            energy_used = 2.0 # kWh
        elif action == 3: # Charge Battery
            if self.battery_level < 100:
                self.battery_level = min(100.0, self.battery_level + 10.0)
                energy_used = 1.0 # kWh
        elif action == 4: # Discharge Battery
            if self.battery_level > 0:
                self.battery_level = max(0.0, self.battery_level - 10.0)
                energy_used = -1.0 # Negative means battery provides power, saving cost

        # 3. Calculate Reward
        # Cost of energy
        cost = energy_used * self.price
        
        # Comfort penalty (squared error from target temperature)
        comfort_penalty = (self.indoor_temp - self.target_temp) ** 2
        
        # Reward is negative of cost and discomfort
        reward = -(cost + 0.1 * comfort_penalty)

        # 4. Update Time and State
        self.current_step += 1
        self.hour = (self.hour + 1) % 24
        self.price = self._get_electricity_price(self.hour)
        
        # Randomly update outdoor temperature
        self.outdoor_temp += np.random.uniform(-0.5, 0.5)
        
        # 5. Check Termination
        done = self.current_step >= self.max_steps
        
        info = {
            "energy_cost": cost,
            "comfort_penalty": comfort_penalty,
            "indoor_temp": self.indoor_temp,
            "battery_level": self.battery_level
        }

        return self.state(), float(reward), done, info

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    @property
    def observation_space(self) -> Box:
        return self._observation_space
