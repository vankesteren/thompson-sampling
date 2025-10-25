from scipy.stats._distn_infrastructure import rv_frozen
import numpy as np

class Arm:
    """
    A single arm in a multi-armed bandit problem.

    Each Arm instance represents a stochastic process that can be "pulled"
    to produce a reward drawn from a specified probability distribution.

    Attributes
    ----------
    id : int
        A unique identifier for the arm.
    dist : rv_frozen
        A frozen SciPy random variable representing the reward distribution
        of the arm (e.g., `scipy.stats.bernoulli(p)` or `scipy.stats.norm(mu, sigma)`).
    """
    def __init__(self, id: int, dist: rv_frozen) -> None:
        """
        Initialize an Arm with a given ID and reward distribution.

        Parameters
        ----------
        id : int
            Unique identifier for this arm.
        dist : rv_frozen
            A frozen random variable (from `scipy.stats`) that defines the
            probability distribution of the arm’s rewards.
        """
        self.id = id
        self.dist = dist

    def pull(self, n: int | None = None) -> int | float | np.ndarray:
        """
        Simulate pulling the arm to generate one or more rewards.

        Parameters
        ----------
        n : int or None, optional
            Number of samples (pulls) to draw. If None, a single reward is drawn.

        Returns
        -------
        int | float | np.ndarray
            A single reward value or an array of rewards, depending on `n`.

        Notes
        -----
        - The type of return value depends on the underlying distribution and the number of samples.
        - For binary bandits, this typically returns 0 or 1.
        """
        return self.dist.rvs(n)

    def __repr__(self) -> str:
        out = f"<Arm {self.id}> "
        out += self.dist.dist.name
        out += str(self.dist.kwds)
        return out

