"""Training utilities for MNIST bandit classification.

Public API
----------
MLPPolicy   : two-layer ReLU MLP
accuracy    : evaluation helper (no_grad)
train_run   : single training run, returns steps / val_accs / test_accs
set_seed    : global RNG seeding
"""

import random
from typing import Any, Dict

import torch
import torch.nn as nn

from methods import ce_loss, pg_loss, dg_loss, entropy_hv_loss, entropy_lv_loss


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MLPPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_sizes: tuple = (50, 50),
        output_dim: int = 10,
    ) -> None:
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(
    model: MLPPolicy,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    chunk: int = 1000,
) -> float:
    model.eval()
    correct = sum(
        (model(x[i : i + chunk].to(device)).argmax(1) == y[i : i + chunk].to(device))
        .sum().item()
        for i in range(0, len(x), chunk)
    )
    return correct / len(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_run(
    method: str,
    seed: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    # model
    hidden_sizes: tuple = (50, 50),
    # optimisation
    batch_size: int = 100,
    train_steps: int = 10_000,
    eval_every: int = 200,
    lr: float = 1e-3,
    # method-specific (unused params are silently ignored)
    eta: float = 1.0,
    beta: float = 0.1,
    baseline_mode: str = "expected",
    samples_per_image: int = 1,
    device: str = "cpu",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a fresh MLPPolicy with *method* and return logged curves.

    Returns a dict with keys:
        steps      : LongTensor  [n_evals+1]   (0, eval_every, 2*eval_every, ...)
        val_accs   : FloatTensor [n_evals+1]
        test_accs  : FloatTensor [n_evals+1]
    """
    set_seed(seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    model = MLPPolicy(hidden_sizes=hidden_sizes).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    steps_log = [0]
    val_accs  = [accuracy(model, x_val,  y_val,  device)]
    test_accs = [accuracy(model, x_test, y_test, device)]

    for step in range(1, train_steps + 1):
        idx = torch.randint(0, len(x_train), (batch_size,), generator=rng)
        xb  = x_train[idx].to(device)
        yb  = y_train[idx].to(device)

        model.train()
        opt.zero_grad()
        logits = model(xb)

        if method == "ce":
            loss = ce_loss(logits, yb)
        elif method == "pg":
            loss = pg_loss(logits, yb, baseline_mode, samples_per_image)
        elif method == "dg":
            loss = dg_loss(logits, yb, eta, baseline_mode, samples_per_image)
        elif method == "entropy_hv":
            loss = entropy_hv_loss(logits, yb, beta, baseline_mode, samples_per_image)
        elif method == "entropy_lv":
            loss = entropy_lv_loss(logits, yb, beta, baseline_mode, samples_per_image)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == train_steps:
            v = accuracy(model, x_val,  y_val,  device)
            t = accuracy(model, x_test, y_test, device)
            steps_log.append(step)
            val_accs.append(v)
            test_accs.append(t)
            if verbose:
                print(f"  step {step:>6}/{train_steps}  val={v:.4f}  test={t:.4f}")

    return {
        "steps":     torch.tensor(steps_log),
        "val_accs":  torch.tensor(val_accs),
        "test_accs": torch.tensor(test_accs),
    }
