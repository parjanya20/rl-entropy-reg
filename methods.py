"""Loss functions for MNIST bandit classification.

Five methods
-----------
ce          : Standard cross-entropy  (full label supervision, oracle baseline)
pg          : Policy Gradient         (REINFORCE with expected-reward baseline)
dg          : Delightful PG           (sigmoid-gated PG, eta=1)
entropy_hv  : Entropy-regularised PG, exact entropy gradient via autodiff
              Loss = PG_loss - beta * H(pi)
              Called _hv because the entropy gradient is computed exactly
              (lower variance for the entropy term relative to entropy_lv).
entropy_lv  : Entropy-regularised PG via score-function estimator (DistG-style)
              weights = (R - b) - beta * (log pi(a|x) + H(pi))
              In expectation the gradient equals that of entropy_hv,
              but the entropy gradient is estimated via sampling (higher variance).

All bandit methods share the same "expected reward" baseline:
    b(x) = E_{a~pi}[R(a)] = sum_a pi(a|x)^2  (for 0/1 rewards)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _baseline(probs: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-sample baseline tensor, shape [batch]."""
    if mode == "expected":
        return (probs ** 2).sum(dim=1)
    if mode == "zero":
        return torch.zeros(probs.shape[0], device=probs.device)
    raise ValueError(f"Unknown baseline mode: {mode!r}")


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy H(pi), shape [batch].  H >= 0."""
    return -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=1)


# ---------------------------------------------------------------------------
# CE
# ---------------------------------------------------------------------------

def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# PG
# ---------------------------------------------------------------------------

def pg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    baseline_mode: str = "expected",
    samples_per_image: int = 1,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    dist  = torch.distributions.Categorical(probs=probs)
    loss  = torch.zeros((), device=logits.device)

    for _ in range(samples_per_image):
        actions = dist.sample()
        log_p   = dist.log_prob(actions)
        rewards = (actions == labels).float()
        adv     = (rewards - _baseline(probs, baseline_mode)).detach()
        loss    = loss - (adv * log_p).mean()

    return loss / samples_per_image


# ---------------------------------------------------------------------------
# DG  (Delightful Policy Gradient)
# ---------------------------------------------------------------------------

def dg_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eta: float = 1.0,
    baseline_mode: str = "expected",
    samples_per_image: int = 1,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    dist  = torch.distributions.Categorical(probs=probs)
    loss  = torch.zeros((), device=logits.device)

    for _ in range(samples_per_image):
        actions   = dist.sample()
        log_p     = dist.log_prob(actions)
        sel_prob  = probs.gather(1, actions.unsqueeze(1)).squeeze(1).clamp_min(1e-9)
        rewards   = (actions == labels).float()
        adv       = rewards - _baseline(probs, baseline_mode)
        surprisal = -sel_prob.log()
        weights   = (torch.sigmoid((adv * surprisal) / eta) * adv).detach()
        loss      = loss - (weights * log_p).mean()

    return loss / samples_per_image


# ---------------------------------------------------------------------------
# Entropy-HV  (exact entropy gradient)
# ---------------------------------------------------------------------------

def entropy_hv_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    baseline_mode: str = "expected",
    samples_per_image: int = 1,
) -> torch.Tensor:
    """PG loss minus beta * H(pi).

    The entropy term differentiates through the softmax exactly,
    so its gradient has lower variance than the score-function estimate used
    in entropy_lv.

    Minimising this loss maximises:  E[R - b] * log pi  +  beta * H(pi)
    """
    probs = torch.softmax(logits, dim=1)
    H     = _entropy(probs).mean()                   # scalar, grad flows through probs
    dist  = torch.distributions.Categorical(probs=probs)
    pg    = torch.zeros((), device=logits.device)

    for _ in range(samples_per_image):
        actions = dist.sample()
        log_p   = dist.log_prob(actions)
        rewards = (actions == labels).float()
        adv     = (rewards - _baseline(probs, baseline_mode)).detach()
        pg      = pg - (adv * log_p).mean()

    return pg / samples_per_image - beta * H


# ---------------------------------------------------------------------------
# Entropy-LV  (score-function entropy estimator, DistG-style)
# ---------------------------------------------------------------------------

def entropy_lv_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    baseline_mode: str = "expected",
    samples_per_image: int = 1,
) -> torch.Tensor:
    """Score-function entropy regularisation (DistG-style).

    Per-sample gradient weights:
        w(a) = (R - b) - beta * (log pi(a|x) + H(pi))

    In expectation over a ~ pi, E[w(a) * grad log pi(a)] equals:
        PG gradient  +  beta * grad H(pi)

    which is identical to entropy_hv in expectation.  The difference is that
    the entropy gradient is estimated via sampling (higher variance).
    """
    probs = torch.softmax(logits, dim=1)
    H     = _entropy(probs)                          # per-sample, shape [batch]
    dist  = torch.distributions.Categorical(probs=probs)
    loss  = torch.zeros((), device=logits.device)

    for _ in range(samples_per_image):
        actions = dist.sample()
        log_p   = dist.log_prob(actions)
        rewards = (actions == labels).float()
        adv     = rewards - _baseline(probs, baseline_mode)
        # Both log_p and H are detached so gradient flows only through the
        # outer log_p in the weighted sum (pure score-function estimator).
        weights = (adv - beta * (log_p.detach() + H.detach())).detach()
        loss    = loss - (weights * log_p).mean()

    return loss / samples_per_image
