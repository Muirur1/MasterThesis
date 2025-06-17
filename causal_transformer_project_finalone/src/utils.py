import pandas as pd
from torch.utils.data import DataLoader
from src.datamodule import TimeSeriesDataset

def summarize_results(results_dict):

    train_metrics = []
    val_metrics = []
    test_metrics = []
    train_step_metrics = []
    val_step_metrics = []
    test_step_metrics = []
    scatter_data = []

    for model_name, metrics in results_dict.items():
        # Epoch-level metrics
        train_df = pd.DataFrame(metrics["train_epochs"])  
        val_df = pd.DataFrame(metrics["val_epochs"])    
        test_df = pd.DataFrame(metrics["test_epochs"])  
        train_df["model"] = model_name
        val_df["model"] = model_name
        test_df["model"] = model_name
        train_metrics.append(train_df)
        val_metrics.append(val_df)
        test_metrics.append(test_df)

        # Step-level metrics - if present
        if len(metrics.get("train_steps", [])) > 0:
            train_step_df = convert_step_summary_wide_to_long(metrics["train_steps"][-1])  # use last epoch
            train_step_df["model"] = model_name
            train_step_metrics.append(train_step_df)

        if len(metrics.get("val_steps", [])) > 0:
            val_step_df = convert_step_summary_wide_to_long(metrics["val_steps"][-1])
            val_step_df["model"] = model_name
            val_step_metrics.append(val_step_df)

        if len(metrics.get("test_steps", [])) > 0:
            test_step_df = convert_step_summary_wide_to_long(metrics["test_steps"][-1])
            test_step_df["model"] = model_name
            test_step_metrics.append(test_step_df)

        # ITE scatter data
        ites = metrics.get("ites", {})
        ite_true = ites.get("test", {}).get("true")
        ite_pred = ites.get("test", {}).get("pred")
        if ite_true is not None and ite_pred is not None:
            scatter_data.append(pd.DataFrame({
                "true_ite": ite_true.flatten(),
                "pred_ite": ite_pred.flatten(),
                "model": model_name
            }))

    return {
        "train_epoch_summary": pd.concat(train_metrics, ignore_index=True),
        "val_epoch_summary": pd.concat(val_metrics, ignore_index=True),
        "test_epoch_summary": pd.concat(test_metrics, ignore_index=True),
        "train_step_summary": pd.concat(train_step_metrics, ignore_index=True) if train_step_metrics else None,
        "val_step_summary": pd.concat(val_step_metrics, ignore_index=True) if val_step_metrics else None,
        "test_step_summary": pd.concat(test_step_metrics, ignore_index=True) if test_step_metrics else None,
        "scatter_summary": pd.concat(scatter_data, ignore_index=True) if scatter_data else None
    }


def convert_epoch_summary_wide_to_long(df):
    return pd.melt(
        df,
        id_vars=["epoch", "model"],
        value_vars=["PEHE", "ATE", "RMSE", "Policy Risk"],
        var_name="metric",
        value_name="value"
    )

def convert_step_summary_wide_to_long(step_summary_dict):
    """
    Convert nested dict {metric: list of values at each timepoint} into long-format DataFrame.

    Args:
        step_summary_dict (dict): e.g., {"PEHE": [...], "ATE": [...], ...}

    Returns:
        pd.DataFrame with columns: ["timepoint", "metric", "value"]
    """
    records = []
    for metric, values in step_summary_dict.items():
        for t, value in enumerate(values):
            records.append({
                "timepoint": t,
                "metric": metric,
                "value": value
            })
    return pd.DataFrame.from_records(records)

def create_dataloaders_from_df(df_train, df_val, covariates, batch_size, outcome_col="pct_weight_gain_factual", treatment_col="binary_treatment"):

    train_set = TimeSeriesDataset(df_train, covariates, outcome_col, treatment_col)
    val_set = TimeSeriesDataset(df_val, covariates, outcome_col, treatment_col)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
