from functools import cached_property

import numpy as np
from scipy.stats.distributions import beta
from tqdm import tqdm

from arm import Arm


class BinomialBandit:
    """
    A class implementing a Bayesian multi-armed bandit algorithm
    using Beta-Binomial Thompson sampling.

    Each arm is assumed to produce binary rewards (0 or 1),
    and the posterior belief about each arm's success probability
    is modeled as a Beta distribution.

    Attributes
    ----------
    arms : list[Arm]
        A list of Arm objects, each representing a bandit arm that can be pulled.
    n_arms : int
        The number of available arms.
    B : np.ndarray
        A (n_arms, 2) array of Beta distribution parameters [alpha, beta]
        for each arm. Initialized to ones.
    eta : float
        A smoothing or exploration parameter added to updates (default is 0.0).
        positive eta -> more greedy, negative eta (but no lower than -1) -> more exploration
    """

    def __init__(self, arms: list[Arm], eta: float = 0.0) -> None:
        """
        Initialize the BinomialBandit with a list of arms and an optional exploration parameter.

        Parameters
        ----------
        arms : list[Arm]
            A list of Arm objects that support a `pull()` method returning a binary reward (0 or 1).
        eta : float, optional
            Additional parameter controlling the strength of updates
            to encourage exploration. Default is 0.0.
            Positive eta -> more exploitation, negative eta (but no lower than -1) -> more exploration
        """
        self.arms = arms
        self.n_arms = len(arms)
        self.B = np.ones((self.n_arms, 2), dtype=np.float64)
        self.eta = eta

    def sample_arm(self, verbose: bool = False):
        """
        Sample an arm index according to Thompson sampling.

        Each arm's success probability is sampled from its current Beta posterior,
        and the arm with the highest sampled probability is chosen.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints the sampled probabilities for all arms.

        Returns
        -------
        int
            The index of the selected arm.
        """
        # TODO: speed this up using https://stats.stackexchange.com/questions/548202/distribution-of-argmax-of-beta-distributed-random-variables
        p_hats = np.array([beta.rvs(a=par[0], b=par[1]) for par in self.B])
        if verbose:
            print(p_hats.round(3))
        return np.argmax(p_hats)

    def update(self, n: int = 1) -> None:
        """
        Perform Bayesian updates to each arm’s Beta parameters
        based on observed rewards from simulated arm pulls.

        For each iteration:
        - An arm is sampled using Thompson sampling.
        - The arm is pulled, and its reward (0 or 1) is observed.
        - The corresponding Beta parameters (α, β) are updated.

        Parameters
        ----------
        n : int, optional
            Number of update iterations to perform. Default is 1.
        """
        for _ in tqdm(range(n)):
            arm_id = self.sample_arm()
            reward = int(self.arms[arm_id].pull())
            self.B[arm_id, 1 - reward] += 1.0 + self.eta

    @property
    def p_hat(self):
        """
        Compute the current mean estimate of success probability for each arm.

        Returns
        -------
        np.ndarray
            An array of estimated success probabilities (posterior means) for each arm.
        """
        return self.B[:, 0] / self.B.sum(axis=1)

    @property
    def pull_count(self):
        """
        Return the total number of times each arm has been pulled.

        This is computed as the sum of the Beta parameters minus their initial values.

        Returns
        -------
        np.ndarray
            An array containing the effective pull counts for each arm.
        """
        return (self.B - 1).sum(axis=1)

    @cached_property
    def p_true(self) -> np.ndarray:
        """
        Ground-truth success probabilities for each arm.

        Extracts the parameter ``p`` from each arm's underlying distribution
        (if available) and falls back to ``0.5`` when missing.

        Returns
        -------
        np.ndarray
            Array of true success probabilities, one per arm.
        """
        return np.array([a.dist.kwds.get("p", 0.5) for a in self.arms])

    @property
    def p_err(self) -> np.ndarray:
        """
        Estimation error for each arm's success probability.

        Defined as ``p_true - p_hat`` for each arm.

        Returns
        -------
        np.ndarray
            Array of per-arm estimation errors.
        """
        return self.p_true - self.p_hat

    @property
    def reward(self) -> float:
        """
        Total number of observed successes across all arms.

        Notes
        -----
        This value is computed from the Beta-parameter matrix ``B`` by summing
        ``alpha - 1`` for all arms, which equals the count of observed successes
        since initialization.

        Returns
        -------
        float
            Cumulative successes over all pulls.
        """
        return (self.B[:, 0] - 1).sum()

    def summary(self) -> None:
        """
        Print a short textual summary of the bandit state.

        The report includes:
        - Total pulls and successes.
        - Mean squared estimation error across arms.
        - Indices and probabilities of the true-best and currently estimated-best arms.
        """
        print(f"[ Bayesian binomial bandit with {self.n_arms} arms ]")
        print(f"- Pulled {int(self.pull_count.sum())} times, with {int(self.reward)} successes.")
        print(f"- Mean squared estimation error: {float((self.p_err * self.p_err).mean()):.3f}")
        i_true = np.argmax(self.p_true)
        i_hat = np.argmax(self.p_hat)
        print(
            f"- true best arm: {i_true}, p_true {self.p_true[i_true]:.3f}, p_hat {self.p_hat[i_true]:.3f}"
        )
        print(
            f"- estd best arm: {i_hat}, p_true {self.p_true[i_hat]:.3f}, p_hat {self.p_hat[i_hat]:.3f}"
        )
