import pandas as pd
import numpy as np

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").drop_duplicates().reset_index(drop=True)
    engine_col = next((c for c in df.columns if c.lower() in ["unit","unit_id","engine_id","engine"]), None)
    cycle_col = next((c for c in df.columns if c.lower() in ["cycle","cycles","time_in_cycles","operating_cycle"]), None)
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    for c in numeric:
        df[c] = df[c].fillna(df[c].median())
    sensor_cols = [c for c in numeric if c not in {engine_col, cycle_col}]
    return df, engine_col, cycle_col, sensor_cols

def summarize_sensors(df, sensor_cols, cycle_col=None):
    out = pd.DataFrame({
        "sensor": sensor_cols,
        "mean": [df[c].mean() for c in sensor_cols],
        "std": [df[c].std() for c in sensor_cols],
        "variance": [df[c].var() for c in sensor_cols],
        "min": [df[c].min() for c in sensor_cols],
        "max": [df[c].max() for c in sensor_cols],
    })
    if cycle_col:
        out["corr_with_cycle"] = [df[[cycle_col,c]].corr().iloc[0,1] for c in sensor_cols]
    return out

def variance_by_age(df, sensor_cols, cycle_col, windows=10):
    bins = pd.qcut(df[cycle_col], q=min(windows, df[cycle_col].nunique()), duplicates="drop")
    rows=[]
    for sensor in sensor_cols:
        vals=df.groupby(bins, observed=True)[sensor].var()
        for interval, value in vals.items():
            rows.append({"sensor":sensor, "cycle_window":str(interval), "variance":value})
    return pd.DataFrame(rows)
