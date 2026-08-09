from analysis import load_and_clean, summarize_sensors, variance_by_age
from viz import plot_sensor_trends, plot_variance_trends

DATA_PATH = "data/raw/train_FD001.csv"

if __name__ == "__main__":
    df, engine_col, cycle_col, sensor_cols = load_and_clean(DATA_PATH)
    summarize_sensors(df, sensor_cols, cycle_col).to_csv(
        "data/cleaned/sensor_summary.csv", index=False
    )
    variance_by_age(df, sensor_cols, cycle_col).to_csv(
        "data/cleaned/windowed_sensor_variance.csv", index=False
    )
    plot_sensor_trends(df, sensor_cols, cycle_col)
    plot_variance_trends(df, sensor_cols, cycle_col)
    print("Analysis completed successfully.")
