import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class TimeSeriesDataset(Dataset):
    def __init__(self, df, covariates, outcome_col, treatment_col, time_col="timepoint", id_col="record_id"):
        self.df = df.copy()
        self.covariates = covariates
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        self.time_col = time_col
        self.id_col = id_col

        # Automatically encode categorical outcome
        self.label_encoder = None
        if df[outcome_col].dtype == "object" or df[outcome_col].dtype.name == "category":
            self.label_encoder = LabelEncoder()
            self.df[outcome_col] = self.label_encoder.fit_transform(self.df[outcome_col])
            # Also encode counterfactual if it exists
            cf_col = outcome_col.replace("factual", "counterfactual")
            if cf_col in df.columns and df[cf_col].notna().all():
                self.df[cf_col] = self.label_encoder.transform(self.df[cf_col])

        # Group by individual
        self.groups = self.df.groupby(id_col)
        self.ids = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        rid = self.ids[idx]
        group = self.groups.get_group(rid).sort_values(by=self.time_col)

        x = torch.tensor(group[self.covariates].values, dtype=torch.float32)
        t = torch.tensor(group[self.treatment_col].values, dtype=torch.float32)
        y = torch.tensor(group[self.outcome_col].values, dtype=torch.float32)
        time = torch.tensor(group[self.time_col].values, dtype=torch.long)
        mask = torch.tensor(group["censored"].values, dtype=torch.bool)

        iptw = torch.tensor(group["iptw"].values, dtype=torch.float32)

        # Handle counterfactuals safely
        y_cf_col = self.outcome_col.replace("factual", "counterfactual")
        if y_cf_col in group.columns:
            y_cf_values = group[y_cf_col].values
            if np.all(np.isnan(y_cf_values)):
                y_cf = torch.zeros_like(y)
            else:
                y_cf_values = np.nan_to_num(y_cf_values, nan=0.0)
                y_cf = torch.tensor(y_cf_values, dtype=torch.float32)
        else:
            y_cf = torch.zeros_like(y)

        # Ensure record_id is Tensor[1]
        rid_tensor = torch.tensor(rid, dtype=torch.long)

        return {
            "x": x,
            "t": t,
            "y": y,
            "y_cf": y_cf,
            "time": time,
            "record_id": rid_tensor,
            "mask": mask,
            "weight": iptw
        }

def create_dataloaders(df: pd.DataFrame,
                       covariates: list,
                       outcome_col: str = "pct_weight_gain_factual",
                       treatment_col: str = "binary_treatment",
                       batch_size: int = 64,
                       test_size: float = 0.4,
                       val_size: float = 0.2,
                       num_workers: int = 4,
                       seed: int = 42):

    np.random.seed(seed)
    record_ids = df["record_id"].unique()
    train_ids, test_ids = train_test_split(record_ids, test_size=test_size, random_state=seed)
    train_ids, val_ids = train_test_split(train_ids, test_size=val_size, random_state=seed)

    df_train = df[df["record_id"].isin(train_ids)]
    df_val = df[df["record_id"].isin(val_ids)]
    df_test = df[df["record_id"].isin(test_ids)]

    train_set = TimeSeriesDataset(df_train, covariates, outcome_col, treatment_col)
    val_set = TimeSeriesDataset(df_val, covariates, outcome_col, treatment_col)
    test_set = TimeSeriesDataset(df_test, covariates, outcome_col, treatment_col)

    return train_set, val_set, test_set, \
           DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers), \
           DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers), \
           DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
