import torch
import torch.nn.functional as F

def mse_loss(y_pred, y_true, mask):
    """
    Mean Squared Error loss with masking.
    Args:
        y_pred: predicted outcomes, shape [B, T]
        y_true: true outcomes, shape [B, T]
        mask: binary mask, shape [B, T], 1 if observed, 0 if censored
    Returns:
        scalar loss
    """
    loss = (y_pred - y_true) ** 2
    return (loss * mask).sum() / mask.sum()

def weighted_mse_loss(y_pred, y_true, mask, weights):
    """
    IPCW-adjusted Mean Squared Error loss with masking.
    Args:
        y_pred: predicted outcomes, shape [B, T]
        y_true: true outcomes, shape [B, T]
        mask: binary mask, shape [B, T]
        weights: Inverse Probability of Censoring Weights, shape [B, T]
    Returns:
        scalar loss
    """
    loss = (y_pred - y_true) ** 2
    return (loss * mask * weights).sum() / (mask * weights).sum()

def cdc_loss(latent_factual, latent_counterfactual):
    """
    Counterfactual Domain Confusion (CDC) loss.

    Encourages alignment between factual and counterfactual latent representations
    by minimizing cosine similarity distance.

    Args:
        latent_factual: tensor of shape [B, T, D]
        latent_counterfactual: tensor of shape [B, T, D]

    Returns:
        scalar loss (1 - cosine similarity)
    """
    # Flatten batch and time for similarity
    lf = latent_factual.reshape(-1, latent_factual.size(-1))  # [B*T, D]
    lc = latent_counterfactual.reshape(-1, latent_counterfactual.size(-1))  # [B*T, D]
    similarity = F.cosine_similarity(lf, lc, dim=-1)
    return 1.0 - similarity.mean()


# Not used[For categorical outcome]
def cross_entropy_loss(y_pred_logits, y_true_labels, mask):
    """
    y_pred_logits: [B, T, C]  logits per class
    y_true_labels: [B, T]     int labels
    mask:          [B, T]     1 if observed
    """
    B, T, C = y_pred_logits.shape
    loss = F.cross_entropy(
        y_pred_logits.view(B * T, C),
        y_true_labels.view(B * T),
        reduction='none'
    )
    mask_flat = mask.view(B * T).float()
    return (loss * mask_flat).sum() / mask_flat.sum()

# Not used[For categorical outcome]
def weighted_cross_entropy_loss(y_pred_logits, y_true_labels, mask, weights):
    """
    IPCW-adjusted Cross-Entropy loss with masking.
    Args:
        y_pred_logits: [B, T, C]
        y_true_labels: [B, T]
        mask:          [B, T]
        weights:       [B, T]
    """
    B, T, C = y_pred_logits.shape
    loss = F.cross_entropy(
        y_pred_logits.view(B * T, C),
        y_true_labels.view(B * T),
        reduction='none'
    )
    mask_flat = mask.view(B * T).float()
    weights_flat = weights.view(B * T)
    return (loss * mask_flat * weights_flat).sum() / (mask_flat * weights_flat).sum()
