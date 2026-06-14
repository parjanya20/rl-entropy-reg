# RL Entropy Regularization

We are studying entropy regularization for policy-gradient estimators. The shared objective is

$$
J_\beta(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[R(\tau)+\beta\sum_t H_\theta(s_t)\right],
\qquad
H_\theta(s)=-\sum_a\pi_\theta(a\mid s)\log\pi_\theta(a\mid s).
$$

For finite samples, the two entropy estimators differ only in how they estimate the entropy-gradient term. With \(N\) sampled state-action positions and advantage estimate \(A_{i,t}\), entropy-HV uses the exact per-state entropy term in the sampled loss:

$$
\widehat{L}_{\mathrm{HV}}=-\frac{1}{N}\sum_{i,t}A_{i,t}\log\pi_\theta(a_{i,t}\mid s_{i,t})-\beta\frac{1}{N}\sum_{i,t}H_\theta(s_{i,t}).
$$

Entropy-LV instead uses the score-function identity for entropy, folding the sampled entropy-gradient contribution into the policy-gradient weight:

$$
\widehat{L}_{\mathrm{LV}}=-\frac{1}{N}\sum_{i,t}\left[A_{i,t}-\beta\left(\log\pi_\theta(a_{i,t}\mid s_{i,t})+H_\theta(s_{i,t})\right)\right]\log\pi_\theta(a_{i,t}\mid s_{i,t}).
$$

One might expect entropy-HV to be the high-variance estimator, since it is the standard entropy-regularized policy-gradient form. Paradoxically, in our runs the opposite happens: the entropy part alone has higher variance for entropy-LV, but the total finite-sample gradient variance is lower.

The experiments follow the MNIST bandit and token-reversal setups from Osband's *Delightful policy gradient* paper [1]. MNIST treats digit classification as a one-step policy-gradient problem, while token reversal trains an autoregressive policy to emit the reversed input sequence, letting us compare REINFORCE, Delightful Policy Gradient, and entropy-regularized variants under the same benchmark structure.

[1] Ian Osband. *Delightful policy gradient*. arXiv preprint arXiv:2603.14608, 2026.
