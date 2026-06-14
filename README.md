# RL Entropy Regularization

Our entropy-LV method trains with a per-token score-function entropy estimator:

$$
L_{\mathrm{LV}}=-\mathbb{E}_t\left[\left(A_t-\beta(\log \pi_\theta(a_t\mid s_t)+H_t)\right)\log \pi_\theta(a_t\mid s_t)\right].
$$

The comparison entropy-HV baseline uses the more obvious exact entropy objective:

$$
L_{\mathrm{HV}}=-\mathbb{E}_t[A_t\log \pi_\theta(a_t\mid s_t)]-\beta\,\mathbb{E}_t[H_t].
$$

One might expect entropy-HV to be the high-variance estimator, since it is the standard entropy-regularized policy-gradient form.
Paradoxically, in our runs the opposite happens: although the entropy contribution itself is noisier for entropy-LV, the total gradient variance is lower.
This makes entropy-LV the method of interest for studying how entropy regularization can reduce overall policy-gradient variance.

The experiments follow the MNIST bandit and token-reversal setups from Osband's *Delightful policy gradient* paper [1]. MNIST treats digit classification as a one-step policy-gradient problem, while token reversal trains an autoregressive policy to emit the reversed input sequence, letting us compare REINFORCE, Delightful Policy Gradient, and entropy-regularized variants under the same benchmark structure.

```bibtex
[1] @article{osband2026delightful,
  title={Delightful policy gradient},
  author={Osband, Ian},
  journal={arXiv preprint arXiv:2603.14608},
  year={2026}
}
```
