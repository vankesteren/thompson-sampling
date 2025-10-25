# Bayesian Beta-Bernoulli multi-armed bandit
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)
![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json) 
![scipy](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/vankesteren/thompson-sampling/main/scipy_button.json)

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
[0.717 0.598 0.713 0.252 0.52  0.628 0.68  0.471 0.478 0.777 0.387 0.551
 0.001 0.268 0.498 0.743 0.682 0.35  0.085 0.522 0.836 0.629 0.039 0.715
 0.51  0.222 0.47  0.451 0.059 0.665 0.7   0.207 0.344 0.992 0.311 0.08
 0.781 0.272 0.864 0.114 0.312 0.064 0.806 0.019 0.372 0.654 0.98  0.674
 0.761 0.233 0.875 0.539 0.744 0.662 0.315 0.904 0.542 0.293 0.539 0.243
 0.898 0.622 0.834 0.089 0.67  0.6   0.058 0.268 0.831 0.481 0.701 0.398
 0.527 0.776 0.989 0.713 0.385 0.355 0.118 0.153 0.448 0.629 0.827 0.051
 0.333 0.139 0.851 0.867 0.724 0.907 0.044 0.633 0.982 0.605 0.148 0.751
 0.795 0.609 0.046 0.834]

Initializing Binomial Bandit

[ Bayesian binomial bandit with 100 arms ]
- Pulled 0 times, with 0 successes.
- Mean squared estimation error: 0.077
- true best arm: 33, p_true 0.992, p_hat 0.500
- estd best arm: 0, p_true 0.717, p_hat 0.500

Updating 250 times...

100%|█████████████████████████████████| 250/250 [00:00<00:00, 268.23it/s]
[ Bayesian binomial bandit with 100 arms ]
- Pulled 250 times, with 172 successes.
- Mean squared estimation error: 0.049
- true best arm: 33, p_true 0.992, p_hat 0.929
- estd best arm: 92, p_true 0.982, p_hat 0.957

Updating another 250 times...

100%|█████████████████████████████████| 250/250 [00:00<00:00, 290.24it/s]
[ Bayesian binomial bandit with 100 arms ]
- Pulled 500 times, with 386 successes.
- Mean squared estimation error: 0.049
- true best arm: 33, p_true 0.992, p_hat 0.951
- estd best arm: 74, p_true 0.989, p_hat 0.970

Let's see if the success probability increases now...

100%|███████████████████████████████| 2500/2500 [00:08<00:00, 298.33it/s]
[ Bayesian binomial bandit with 100 arms ]
- Pulled 3000 times, with 2820 successes.
- Mean squared estimation error: 0.043
- true best arm: 33, p_true 0.992, p_hat 0.990
- estd best arm: 33, p_true 0.992, p_hat 0.990
```