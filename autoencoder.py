import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

def run_autoencoder(path="data/cleaned/sensor_data_cleaned.csv",
                    output="data/cleaned/autoencoder_anomaly_scores.csv"):
    df=pd.read_csv(path)
    sensor_cols=[c for c in df.select_dtypes(include=np.number).columns
                 if c not in ["engine_id","cycle"]]

    # Treat the earliest 20% of each engine's cycles as a healthy reference.
    parts=[]
    for _,g in df.groupby("engine_id"):
        g=g.sort_values("cycle")
        parts.append(g.iloc[:max(1,int(len(g)*0.20))])
    healthy=pd.concat(parts,ignore_index=True)

    scaler=StandardScaler()
    X_train=scaler.fit_transform(healthy[sensor_cols])
    X_all=scaler.transform(df[sensor_cols])

    model=MLPRegressor(hidden_layer_sizes=(32,16,32), activation="relu",
                       solver="adam", learning_rate_init=0.001,
                       max_iter=250, early_stopping=True,
                       validation_fraction=0.1, random_state=42)
    model.fit(X_train,X_train)

    reconstructed=model.predict(X_all)
    errors=np.mean((X_all-reconstructed)**2,axis=1)

    healthy_reconstructed=model.predict(X_train)
    healthy_errors=np.mean((X_train-healthy_reconstructed)**2,axis=1)
    threshold=np.percentile(healthy_errors,95)

    out=df[["engine_id","cycle"]].copy()
    out["reconstruction_error"]=errors
    out["anomaly"]=errors>threshold
    out.to_csv(output,index=False)

    return out, threshold
