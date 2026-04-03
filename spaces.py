from typing import List, Tuple, Union
import numpy as np

class Space:
    """Base class for environment spaces."""
    def sample(self):
        raise NotImplementedError

    def contains(self, x):
        raise NotImplementedError

class Discrete(Space):
    """A discrete space with N possible actions (0, 1, ..., n-1)."""
    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return np.random.randint(self.n)

    def contains(self, x) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n

    def __repr__(self):
        return f"Discrete({self.n})"

class Box(Space):
    """A continuous space defined by lower and upper bounds."""
    def __init__(self, low: Union[float, np.ndarray], high: Union[float, np.ndarray], shape: Tuple[int, ...] = None):
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = shape or self.low.shape

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=self.shape)

    def contains(self, x) -> bool:
        return np.all(x >= self.low) and np.all(x <= self.high)

    def __repr__(self):
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"
