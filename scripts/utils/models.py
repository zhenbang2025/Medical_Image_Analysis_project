import torch
import torch.nn as nn


class BBoxMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2, arch: str = "plain"):
        super().__init__()
        self.arch = arch
        if arch == "plain":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 4),
            )
        elif arch == "residual":
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.res_block = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            self.skip_proj = nn.Linear(hidden_dim, hidden_dim)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 4),
            )
        else:
            raise ValueError(f"Unsupported arch: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch == "plain":
            return self.net(x)
        h = self.input_proj(x)
        h = self.res_block(h) + self.skip_proj(h)
        return self.head(h)
