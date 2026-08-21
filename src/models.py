"""CNN architectures for the Vision Robustness Benchmark."""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1),
            nn.Flatten(),
            nn.Linear(15376, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StrongerCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


_MODELS = {
    "simple": SimpleCNN,
    "stronger": StrongerCNN,
}


def build_model(name: str) -> nn.Module:
    try:
        model_cls = _MODELS[name]
    except KeyError:
        raise ValueError(
            f"Unknown model name {name!r}; expected one of {sorted(_MODELS)}"
        ) from None
    return model_cls()

def load_model(name: str, load_path: str, device = 'cpu') -> nn.Module:
    model = build_model(name)
    model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
    return model
