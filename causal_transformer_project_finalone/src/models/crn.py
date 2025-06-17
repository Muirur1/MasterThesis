import torch
import torch.nn as nn

class CRN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.1, use_ipcw=False, use_cdc=True):
        super().__init__()
        self.use_ipcw = use_ipcw
        self.use_cdc = use_cdc

        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )

        self.head_y0 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )
        self.head_y1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, t, mask=None):
        h_out, h_n = self.gru(x)  # [B, T, H]

        y0_hat = self.head_y0(h_out).squeeze(-1)  # [B, T]
        y1_hat = self.head_y1(h_out).squeeze(-1)  # [B, T]

        y_factual = torch.where(t == 1, y1_hat, y0_hat)
        y_cf_pred = torch.where(t == 1, y0_hat, y1_hat)

        output = {
            "factual_pred": y_factual,
            "counterfactual_pred": y_cf_pred,
        }

        if self.use_cdc:
            output["latent_factual"] = h_out
            output["latent_counterfactual"] = h_out

        return output
