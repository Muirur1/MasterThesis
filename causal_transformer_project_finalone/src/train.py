import torch
from src.metrics import compute_all_metrics_full
from src.losses import mse_loss, weighted_mse_loss, cdc_loss
from src.losses import cross_entropy_loss, weighted_cross_entropy_loss
import pandas as pd
from tqdm import trange
import numpy as np

def compute_step_metrics(ite_true, ite_pred, y_true, y_pred, weight):
    n_time = ite_true.shape[1]
    step_metrics = {"PEHE": [], "ATE": [], "RMSE": [], "Policy Risk": []}

    for t in range(n_time):
        pehe_t, ate_t, rmse_t, policy_risk_t = compute_all_metrics_full(
            ite_true[:, t], ite_pred[:, t],
            y_true[:, t], y_pred[:, t],
            weight[:, t],
            return_dict=False
        )
        step_metrics["PEHE"].append(pehe_t)
        step_metrics["ATE"].append(ate_t)
        step_metrics["RMSE"].append(rmse_t)
        step_metrics["Policy Risk"].append(policy_risk_t)

    return step_metrics

def training_loop(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    loss_type="mse",
    use_ipcw=False,
    use_cdc=False,
    lambda_cdc=0.1,
    device="cpu",
    epochs=50
):
    metrics_dict = {
        "train_steps": [], "val_steps": [], "test_steps": [],
        "train_epochs": [], "val_epochs": [], "test_epochs": [],
        "train_loss": [], "val_loss": [], "test_loss": [],
        "train_loss_steps": [], "val_loss_steps": [], "test_loss_steps": [],
        "final_metrics": {},
        "ites": {}
    }

    for epoch in trange(epochs, desc="Training Epochs"):
        model.train()
        train_preds, train_trues, train_cf_trues, train_weights, train_masks, train_ites, train_cf_preds = [], [], [], [], [], [], []
        epoch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()

            # Only real tensors: move to device
            x, t, y, y_cf, time, mask = [batch[k].to(device) for k in ["x", "t", "y", "y_cf", "time", "mask"]]
            record_id = batch["record_id"]  # integer, leave on CPU
            iptw = batch["weight"].to(device)

            weights = iptw * mask.float() if use_ipcw else iptw

            out = model(x, t, mask=mask)
            y_pred = out["factual_pred"]
            y_cf_pred = out["counterfactual_pred"]

            if loss_type == "mse":
                loss = weighted_mse_loss(y_pred, y, mask, weights)
            elif loss_type == "ce":
                loss = weighted_cross_entropy_loss(y_pred, y.long(), mask, weights)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

            latent_factual = out.get("latent_factual", None)
            latent_counterfactual = out.get("latent_counterfactual", None)
            if use_cdc and latent_factual is not None and latent_counterfactual is not None:
                cdc = cdc_loss(latent_factual, latent_counterfactual)
                loss += lambda_cdc * cdc

            epoch_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            train_preds.append(y_pred.detach().cpu().numpy())
            train_cf_preds.append(y_cf_pred.detach().cpu().numpy())
            train_trues.append(y.detach().cpu().numpy())
            train_cf_trues.append(y_cf.detach().cpu().numpy())
            train_ites.append((y_cf - y).detach().cpu().numpy())
            train_weights.append(weights.detach().cpu().numpy())
            train_masks.append(mask.detach().cpu().numpy())

        # Flatten arrays
        y_true_all = np.concatenate(train_trues)
        y_cf_true_all = np.concatenate(train_cf_trues)
        y_pred_all = np.concatenate(train_preds)
        y_cf_pred_all = np.concatenate(train_cf_preds)
        ite_true_all = np.concatenate(train_ites)
        ite_pred_all = y_cf_pred_all - y_pred_all
        weights_all = np.concatenate(train_weights)

        train_metrics_step = compute_all_metrics_full(
            ite_true_all, ite_pred_all, y_true_all, y_pred_all, weights_all, return_dict=True
        )
        metrics_dict["train_epochs"].append({"epoch": epoch, **train_metrics_step})
        metrics_dict["ites"]["train"] = {"true": ite_true_all, "pred": ite_pred_all}

        step_metrics_epoch = compute_step_metrics(ite_true_all, ite_pred_all, y_true_all, y_pred_all, weights_all)
        metrics_dict["train_steps"].append(step_metrics_epoch)

        metrics_dict["train_loss"].append(np.mean(epoch_losses))
        metrics_dict["train_loss_steps"].append(step_metrics_epoch["RMSE"])

        # Evaluate val/test
        def evaluate(loader, name):
            model.eval()
            preds, trues, cf_trues, ites, cf_preds, weights, masks = [], [], [], [], [], [], []
            batch_losses = []

            with torch.no_grad():
                for batch in loader:
                    x, t, y, y_cf, time, mask = [batch[k].to(device) for k in ["x", "t", "y", "y_cf", "time", "mask"]]
                    record_id = batch["record_id"]
                    iptw = batch["weight"].to(device)

                    weights_eval = iptw * mask.float() if use_ipcw else iptw

                    out = model(x, t, mask=mask)
                    y_pred = out["factual_pred"]
                    y_cf_pred = out["counterfactual_pred"]

                    if loss_type == "mse":
                        loss = weighted_mse_loss(y_pred, y, mask, weights_eval)
                    elif loss_type == "ce":
                        loss = weighted_cross_entropy_loss(y_pred, y.long(), mask, weights_eval)
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")

                    latent_factual = out.get("latent_factual", None)
                    latent_counterfactual = out.get("latent_counterfactual", None)
                    if use_cdc and latent_factual is not None and latent_counterfactual is not None:
                        cdc = cdc_loss(latent_factual, latent_counterfactual)
                        loss += lambda_cdc * cdc

                    batch_losses.append(loss.item())

                    preds.append(y_pred.cpu().numpy())
                    cf_preds.append(y_cf_pred.cpu().numpy())
                    trues.append(y.cpu().numpy())
                    cf_trues.append(y_cf.cpu().numpy())
                    ites.append((y_cf - y).cpu().numpy())
                    weights.append(weights_eval.cpu().numpy())

            y_true_all = np.concatenate(trues)
            y_cf_true_all = np.concatenate(cf_trues)
            y_pred_all = np.concatenate(preds)
            y_cf_pred_all = np.concatenate(cf_preds)
            ite_true_all = np.concatenate(ites)
            ite_pred_all = y_cf_pred_all - y_pred_all
            weight_all = np.concatenate(weights)

            metrics = compute_all_metrics_full(
                ite_true_all, ite_pred_all, y_true_all, y_pred_all, weight_all, return_dict=True
            )
            metrics_dict[f"{name}_epochs"].append({"epoch": epoch, **metrics})
            metrics_dict["ites"][name] = {"true": ite_true_all, "pred": ite_pred_all}

            step_metrics_epoch = compute_step_metrics(ite_true_all, ite_pred_all, y_true_all, y_pred_all, weight_all)
            metrics_dict[f"{name}_steps"].append(step_metrics_epoch)

            metrics_dict[f"{name}_loss"].append(np.mean(batch_losses))
            metrics_dict[f"{name}_loss_steps"].append(step_metrics_epoch["RMSE"])

        evaluate(val_loader, "val")
        evaluate(test_loader, "test")

        print(f"Epoch {epoch}: Train PEHE={train_metrics_step['PEHE']:.4f}, ATE={train_metrics_step['ATE']:.4f}, RMSE={train_metrics_step['RMSE']:.4f}")

    # Final metrics
    final_train = metrics_dict["train_epochs"][-1]
    final_val = metrics_dict["val_epochs"][-1]
    final_test = metrics_dict["test_epochs"][-1]

    print(f"\n>>> FINAL TEST METRICS (EPOCH {epoch}):")
    print(f"Test PEHE={final_test['PEHE']:.4f}, ATE={final_test['ATE']:.4f}, "
        f"RMSE={final_test['RMSE']:.4f}, Policy Risk={final_test['Policy Risk']:.4f}")

    metrics_dict["final_metrics"] = {
        "Final Train PEHE": final_train["PEHE"],
        "Final Val PEHE": final_val["PEHE"],
        "Final Test PEHE": final_test["PEHE"],
        "Final Train ATE": final_train["ATE"],
        "Final Val ATE": final_val["ATE"],
        "Final Test ATE": final_test["ATE"],
        "Final Train RMSE": final_train["RMSE"],
        "Final Val RMSE": final_val["RMSE"],
        "Final Test RMSE": final_test["RMSE"],
        "Final Train Policy Risk": final_train["Policy Risk"],
        "Final Val Policy Risk": final_val["Policy Risk"],
        "Final Test Policy Risk": final_test["Policy Risk"],
    }

    return metrics_dict
