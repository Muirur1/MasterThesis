import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class CausalTransformer(nn.Module):
    def __init__(
        self, input_dim, embed_dim=128, num_layers=2, n_heads=4, dropout=0.1,
        use_ipcw=False, use_cdc=False, num_classes=3 
    ):
        super().__init__()
        self.use_ipcw = use_ipcw
        self.use_cdc = use_cdc
        self.num_classes = num_classes

        # Tell the model at training time which head to use (MSE or CE)
        self._loss_type = "mse"  # default

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression heads (for MSE)
        self.head_y0 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1)
        )
        self.head_y1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1)
        )

        # Classification heads (for CE)
        self.head_y0_logits = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, num_classes)
        )
        self.head_y1_logits = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x, t, mask=None):
        """
        Args:
            x:     [B, T, F]   - input covariates
            t:     [B, T]      - binary treatment indicators (0 or 1)
            mask:  [B, T]      - 1 if observed, 0 if censored (optional)

        Returns:
            Dictionary with:
                - factual_pred
                - counterfactual_pred
                - latent_factual, latent_counterfactual (if use_cdc=True)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        pad_mask = ~mask.bool() if mask is not None else None
        h = self.transformer_encoder(x, src_key_padding_mask=pad_mask)

        # CASE 1: Classification (CE)
        if self.training and hasattr(self, "_loss_type") and self._loss_type == "ce":
            y0_logits = self.head_y0_logits(h)  # [B, T, C]
            y1_logits = self.head_y1_logits(h)  # [B, T, C]

            # Expand t for selecting logits
            t_exp = t.unsqueeze(-1).repeat(1, 1, y0_logits.size(-1))  # [B, T, C]

            y_factual_logits = torch.where(t_exp == 1, y1_logits, y0_logits)
            y_counterfactual_logits = torch.where(t_exp == 1, y0_logits, y1_logits)

            output = {
                "factual_pred": y_factual_logits,            # logits
                "counterfactual_pred": y_counterfactual_logits,
            }

        # CASE 2: Regression (MSE)
        else:
            y0_hat = self.head_y0(h).squeeze(-1)  # [B, T]
            y1_hat = self.head_y1(h).squeeze(-1)  # [B, T]

            y_factual = torch.where(t == 1, y1_hat, y0_hat)
            y_cf_pred = torch.where(t == 1, y0_hat, y1_hat)

            output = {
                "factual_pred": y_factual,
                "counterfactual_pred": y_cf_pred,
            }

        # Add latent if using CDC
        if self.use_cdc:
            output["latent_factual"] = h
            output["latent_counterfactual"] = h

        return output
