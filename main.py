import numpy as np
from scipy.stats.distributions import bernoulli

from arm import Arm
from bandit import BinomialBandit

def main():
    
    print("Initializing 100 arms")
    n_arms = 100
    p_true = np.random.uniform(size=n_arms)
    
    print(f"True probabilities:\n{p_true.round(3)}\n")
    arms = [Arm(id=i, dist=bernoulli(p=prob)) for i, prob in enumerate(p_true)]

    print("Initializing Binomial Bandit\n")
    bb = BinomialBandit(arms)
    bb.summary()

    print("\nUpdating 250 times...\n")
    bb.update(250)
    bb.summary()
    
    print("\nUpdating another 250 times...\n")
    bb.update(250)
    bb.summary()

    print("\nLet's see if the success probability increases now... \n")
    bb.update(2500)
    bb.summary()


if __name__ == "__main__":
    main()
