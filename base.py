from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
from schema import Observation, Action, Reward

class OpenEnv(ABC):
    """
    Standard interface for OpenEnv environments using Pydantic models.
    """

    @abstractmethod
    def reset(self, seed: int = None) -> Observation:
        """
        Resets the environment to its initial state.
        Returns the initial Observation model.
        """
        pass

    @abstractmethod
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Takes a Pydantic Action and returns Observation, Reward, done, and info.
        """
        pass

    @abstractmethod
    def state(self) -> Observation:
        """
        Returns the current state as an Observation model.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Returns environment metadata.
        """
        pass
