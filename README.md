# Bayesian Beta-Bernoulli multi-armed bandit
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)
![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)

This is a short & simple exploration of a Bayesian bandit with multiple arms with a Bernoulli reward.

## Installation

Install [uv](https://docs.astral.sh/uv), and then run:

```sh
uv sync
```

from this repository.

## Running

Simply run 

```sh
uv run main.py
``` 

which will run the sample program to produce the following (or similar) output:

```
Initializing 100 arms
True probabilities:
[0.3   0.244 0.435 0.703 0.261 0.354 0.275 0.492 0.858 0.305 0.395 0.638
 0.512 0.344 0.981 0.978 0.065 0.807 0.964 0.424 0.07  0.152 0.473 0.515
 0.278 0.345 0.693 0.543 0.427 0.552 0.983 0.849 0.691 0.794 0.619 0.175
 0.918 0.413 0.436 0.704 0.942 0.745 0.409 0.369 0.637 0.808 0.029 0.364
 0.525 0.942 0.785 0.335 0.792 0.074 0.752 0.16  0.694 0.74  0.676 0.287
 0.048 0.12  0.857 0.406 0.547 0.682 0.607 0.858 0.332 0.61  0.326 0.559
 0.358 0.266 0.304 0.21  0.648 0.259 0.225 0.961 0.762 0.422 0.708 0.298
 0.776 0.053 0.425 0.783 0.538 0.849 0.202 0.749 0.983 0.994 0.584 0.955
 0.767 0.259 0.176 0.558]

Initializing Binomial Bandit

[ Bayesian binomial bandit with 100 arms ]
- Pulled 0 times: 0 successes & 0 failures
- Mean squared estimation error: 0.073
- true best arm: 93, p_true 0.994, p_hat 0.500
- estd best arm: 0, p_true 0.300, p_hat 0.500

Updating 250 times...

100%|█████████████████████████████████████| 250/250 [00:00<00:00, 6099.71it/s]

[ Bayesian binomial bandit with 100 arms ]
- Pulled 250 times: 180 successes & 70 failures
- Regret: 68.528 (proportion: 0.274)
- Mean squared estimation error: 0.046
- true best arm: 93, p_true 0.994, p_hat 0.955
- estd best arm: 93, p_true 0.994, p_hat 0.955

Updating another 250 times...

100%|█████████████████████████████████████| 250/250 [00:00<00:00, 5807.32it/s]

[ Bayesian binomial bandit with 100 arms ]
- Pulled 500 times: 409 successes & 91 failures
- Regret: 88.055 (proportion: 0.176)
- Mean squared estimation error: 0.039
- true best arm: 93, p_true 0.994, p_hat 0.989
- estd best arm: 93, p_true 0.994, p_hat 0.989

Let's see if the success probability increases now...

100%|███████████████████████████████████| 2500/2500 [00:00<00:00, 5821.15it/s]

[ Bayesian binomial bandit with 100 arms ]
- Pulled 3000 times: 2870 successes & 130 failures
- Regret: 112.331 (proportion: 0.037)
- Mean squared estimation error: 0.037
- true best arm: 93, p_true 0.994, p_hat 0.995
- estd best arm: 93, p_true 0.994, p_hat 0.995
```