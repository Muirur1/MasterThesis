import numpy as np
import pandas as pd

def prepare_real_data(df_trial: pd.DataFrame, expected_timepoints: int = 4) -> pd.DataFrame:
    """
    Align real longitudinal data (df_trial) with the simulated data format:
    - Renames factual outcomes
    - Adds counterfactual placeholders
    - Computes censoring indicators
    - Adds dummy ITE columns
    - Computes timepoint_censored
    - Ensures consistent datatypes for modeling
    """
    df_real = df_trial.copy()

    #  FORCE record_id to int64: very important
    df_real["record_id"] = df_real["record_id"].astype(np.int64)

    # Rename outcomes to match simulated naming
    df_real.rename(columns={
        "pct_weight_gain": "pct_weight_gain_factual",
        "feed_outcome": "feed_outcome_factual"
    }, inplace=True)

    # Add placeholder counterfactuals
    df_real["pct_weight_gain_counterfactual"] = np.nan
    df_real["feed_outcome_counterfactual"] = np.nan

    # Add dummy ITE columns
    df_real["ite_weight"] = np.nan
    df_real["ite_feed"] = np.nan

    # Censoring logic
    visit_schedule = [0, 45, 90, 180]  # Explicitly define visit schedule for real data

    visit_counts = df_real[df_real["missed_visit"] == 0].groupby("record_id")["timepoint"].count()
    last_valid_timepoint = df_real[df_real["missed_visit"] == 0].groupby("record_id")["timepoint"].max()

    # If patient completed expected_timepoints: not censored (0), otherwise censored (1)
    df_real["censored"] = df_real["record_id"].map(
        lambda rid: 0 if visit_counts.get(rid, 0) == expected_timepoints else 1
    ).astype(np.int64)

    # Compute timepoint_censored properly
    def get_censoring_time(rid):
        if visit_counts.get(rid, 0) == expected_timepoints:
            return np.nan  # Not censored
        last_t = last_valid_timepoint.get(rid, np.nan)
        future_visits = [v for v in visit_schedule if v > last_t]
        return future_visits[0] if len(future_visits) > 0 else np.nan

    df_real["timepoint_censored"] = df_real["record_id"].map(get_censoring_time)

    # Force consistent datatypes (targeting your list)

    dtype_overrides = {
        "record_id": "int64",
        "timepoint": "int64",
        "agemons": "float64",
        "binary_treatment": "int64",
        "pscore": "float64",
        "iptw": "float64",
        "missed_visit": "int64",
        "censored": "int64",
        "timepoint_censored": "float64",
        "ite_weight": "float64",
        "ite_feed": "float64",
        "pct_weight_gain_factual": "float64",
        "pct_weight_gain_counterfactual": "float64",
    }

    for col, dtype in dtype_overrides.items():
        if col in df_real.columns:
            df_real[col] = df_real[col].astype(dtype)

    # Return fully consistent df_real
    return df_real
