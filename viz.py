import os
import matplotlib.pyplot as plt

def plot_sensor_trends(df, sensor_cols, cycle_col, output="data/cleaned/sensor_trends.png"):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plt.figure(figsize=(10,6))
    for c in sensor_cols[:6]:
        s=df.groupby(cycle_col)[c].mean()
        z=(s-s.mean())/(s.std() if s.std()!=0 else 1)
        plt.plot(z.index, z.values, label=c)
    plt.xlabel("Operating Cycle")
    plt.ylabel("Standardized Mean Sensor Value")
    plt.title("Sensor Trends vs Operating Cycle")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()

def plot_variance_trends(df, sensor_cols, cycle_col, output="data/cleaned/variance_trends.png"):
    import pandas as pd
    import numpy as np
    os.makedirs(os.path.dirname(output), exist_ok=True)
    bins=pd.qcut(df[cycle_col], q=min(10, df[cycle_col].nunique()), duplicates="drop")
    plt.figure(figsize=(10,6))
    for c in sensor_cols[:6]:
        v=df.groupby(bins, observed=True)[c].var()
        plt.plot(range(len(v)), v.values, marker="o", label=c)
    plt.xlabel("Increasing Equipment-Age Window")
    plt.ylabel("Sensor Variance")
    plt.title("Sensor Variance Across Equipment-Age Windows")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
