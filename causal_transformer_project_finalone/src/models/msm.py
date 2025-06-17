import torch
import torch.nn as nn

class MSM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1, use_ipcw=False, use_cdc=False):
        super().__init__()
        self.use_ipcw = use_ipcw
        self.use_cdc = use_cdc

        # Process each time step independently: linear model
        self.head_y0 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.head_y1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, t, mask=None):
        # x - [B, T, F]
        B, T, F = x.shape

        x_flat = x.view(B * T, F)

        y0_hat = self.head_y0(x_flat).view(B, T)
        y1_hat = self.head_y1(x_flat).view(B, T)

        y_factual = torch.where(t == 1, y1_hat, y0_hat)
        y_cf_pred = torch.where(t == 1, y0_hat, y1_hat)

        output = {
            "factual_pred": y_factual,
            "counterfactual_pred": y_cf_pred,
        }

        return output
