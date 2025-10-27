import numpy as np

class Arm:
    """
    A single binomial arm in a multi-armed bandit problem.

    Each Arm instance represents a stochastic process that can be "pulled"
    to produce a reward drawn from a binomial trial.

    Attributes
    ----------
    id : int
        A unique identifier for the arm.
    prob : float
        The probability of success (reward 1)
    """
    def __init__(self, id: int, prob: float) -> None:
        """
        Initialize an Arm with a given ID and reward distribution.

        Parameters
        ----------
        id : int
            Unique identifier for this arm.
        prob : float
            Number between 0 and 1 indicating the probability of success
        """
        self.id = id
        self.prob = prob

    def pull(self, n: int = 1) -> float:
        """
        Simulate pulling the arm to generate one or more rewards.

        Parameters
        ----------
        n : int, default 1
            Number of samples (pulls) to draw.

        Returns
        -------
        float
            A single reward value.
        """
        return np.random.binomial(n, self.prob)

    def __repr__(self) -> str:
        return f"Arm {self.id}, p: {self.prob}"

