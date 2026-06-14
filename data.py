"""MNIST data loading with train / val / test split.

Split sizes (defaults):
    train : 59 000
    val   :  1 000  (carved from the 60 k training set)
    test  : 10 000  (the standard MNIST test set, never used for hparam selection)
"""

from typing import Tuple

import torch
from torchvision import datasets, transforms


def load_mnist_split(
    val_size: int = 1000,
    seed: int = 42,
    data_root: str = "./data",
) -> Tuple[torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Return (x_train, y_train, x_val, y_val, x_test, y_test) as CPU tensors.

    The val split is drawn from the 60 k training pool using a deterministic
    permutation seeded by *seed*.  The test set is the standard MNIST test set.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=data_root, train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    x_all = train_ds.data.float().view(-1, 784) / 255.0
    y_all = train_ds.targets.long()

    x_test = test_ds.data.float().view(-1, 784) / 255.0
    y_test = test_ds.targets.long()

    rng  = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(x_all), generator=rng)

    val_idx   = perm[:val_size]
    train_idx = perm[val_size:]

    return (
        x_all[train_idx], y_all[train_idx],
        x_all[val_idx],   y_all[val_idx],
        x_test,           y_test,
    )
