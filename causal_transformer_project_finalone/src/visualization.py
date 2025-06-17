import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_epoch_summary(df_long, split):
    """
    Plot PEHE, ATE, RMSE, Policy Risk per epoch by model for a given split.
    One row of subplots with common legend.
    """
    df_split = df_long[df_long["split"] == split]
    metrics = df_split["metric"].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), sharey=False)

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        df_plot = df_split[df_split["metric"] == metric]
        sns.lineplot(data=df_plot, x="epoch", y="value", hue="model", marker="o", ax=axes[i])
        axes[i].set_title(f"{split} {metric} per Epoch")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    for ax in axes:
        ax.get_legend().remove()

    plt.tight_layout()
    plt.show()

def plot_step_summary(df_long, split):
    """
    Plot PEHE, ATE, RMSE, Policy Risk per timepoint by model for a given split.
    One row of subplots with common legend.
    """
    df_split = df_long[df_long["split"] == split]
    metrics = df_split["metric"].unique()

    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5), sharey=False)

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        df_plot = df_split[df_split["metric"] == metric]
        sns.lineplot(data=df_plot, x="timepoint", y="value", hue="model", marker="o", ax=axes[i])
        axes[i].set_title(f"{split} {metric} per Timepoint")
        axes[i].set_xlabel("Timepoint")
        axes[i].set_ylabel(metric)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    for ax in axes:
        ax.get_legend().remove()

    plt.tight_layout()
    plt.show()

def plot_loss_curves(loss_dict, split="Train"):
    """
    Plot loss per epoch and per timepoint.
    """
    # Loss per epoch
    plt.figure(figsize=(8, 5))
    plt.plot(loss_dict["epoch"], marker="o")
    plt.title(f"{split} Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Loss per timepoint (last epoch)
    last_timepoint_losses = loss_dict["per_timepoint"][-1] if isinstance(loss_dict["per_timepoint"][0], (list, np.ndarray)) else loss_dict["per_timepoint"]
    plt.figure(figsize=(8, 5))
    plt.plot(last_timepoint_losses, marker="o")
    plt.title(f"{split} Loss per Timepoint (Final Epoch)")
    plt.xlabel("Timepoint")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ite_scatter(true_ite, pred_ite, model_name="Model"):
    """
    Plot predicted vs true ITE scatter plot.
    """
    assert true_ite.shape == pred_ite.shape, "Mismatched ITE shapes"

    true_flat = true_ite.flatten()
    pred_flat = pred_ite.flatten()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=true_flat, y=pred_flat, alpha=0.3)
    max_val = max(np.max(true_flat), np.max(pred_flat))
    min_val = min(np.min(true_flat), np.min(pred_flat))
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    plt.xlabel("True ITE")
    plt.ylabel("Predicted ITE")
    plt.title(f"ITE Scatter Plot: {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loss_curves_all_models(loss_dict_all_models, split="Train"):
    """
    Plot loss per epoch and per timepoint for ALL MODELS.
    One row with 2 subplots: Loss per Epoch, Loss per Timepoint.
    """
    models = list(loss_dict_all_models.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Plot Loss per Epoch
    for model in models:
        epoch_loss = loss_dict_all_models[model]["epoch"]
        axes[0].plot(epoch_loss, marker="o", label=model)
    axes[0].set_title(f"{split} Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # Plot Loss per Timepoint (last epoch)
    for model in models:
        per_tp = loss_dict_all_models[model]["per_timepoint"]
        last_tp_loss = per_tp[-1] if isinstance(per_tp[0], (list, np.ndarray)) else per_tp
        axes[1].plot(last_tp_loss, marker="o", label=model)
    axes[1].set_title(f"{split} Loss per Timepoint (Final Epoch)")
    axes[1].set_xlabel("Timepoint")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)

    # Common Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
