# Smart Manufacturing: Sensor Drift in Production Lines

## Overview
This project analyzes NASA C-MAPSS turbofan engine sensor logs to investigate whether sensor drift and variability change as equipment operating age increases.

## Research Question
Does sensor variability and drift increase with operating cycles, and can these trends provide useful early indicators of equipment degradation without requiring massive labeled failure datasets?

## Structure
- `data/raw/` - original dataset
- `data/cleaned/` - cleaned dataset and analysis outputs
- `src/` - Python analysis and visualization scripts
- `notebooks/` - exploratory Jupyter notebook
- `docs/` - report, presentation, and video demo

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/main.py
```

## Analysis
The project performs data cleaning, sensor statistics, trend analysis, variance-by-age analysis, and visualization.

## Future Scope
A future version can use autoencoders for unsupervised anomaly detection and reconstruction-error-based maintenance alerts.

## Dataset Results
- Rows: 20,631
- Columns: 26
- Engines: 100
- Sensor/measurement columns analyzed: 24

## Preliminary findings
The strongest absolute linear relationships with operating cycle are:
   sensor  correlation_with_cycle
sensor_11                0.634385
 sensor_4                0.624577
sensor_12               -0.611354
 sensor_7               -0.595914
sensor_15                0.588676

The sensors with the strongest absolute increase/decrease in windowed variance are:
   sensor  variance_slope
 sensor_9        5.735227
sensor_14        4.107370
 sensor_4        0.704015
 sensor_3        0.194080
sensor_17        0.022323

These are screening results, not proof of causation. Further statistical testing and validation on held-out engines are recommended.


## Autoencoder Anomaly Detection
An unsupervised reconstruction-based anomaly detector has been added in `src/autoencoder.py`.

### Approach
1. Use the earliest 20% of each engine's cycles as a healthy reference.
2. Standardize all sensor features.
3. Train an MLP autoencoder-like reconstruction model.
4. Calculate reconstruction error for every observation.
5. Set the anomaly threshold at the 95th percentile of healthy-reference reconstruction error.
6. Flag observations above the threshold as potential anomalies.

The model is a baseline for demonstration and should be validated more rigorously before real maintenance decisions.
