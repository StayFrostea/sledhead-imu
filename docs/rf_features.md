# Random Forest Feature Extraction

## Overview

Comprehensive feature extraction from raw IMU data for Random Forest classification. This module extracts 22 features per run from accelerometer and gyroscope data sampled at 2000 Hz.

**Important**: Input data must already be in **g units** (normalized to standard gravity: 1g = 9.80665 m/s²). If your raw data is in m/s², convert first using: `accel_g = accel_mps2 / 9.80665`.

## Features Extracted

### 1. Basic Statistics (Acceleration)

- `accel_mean`: Mean resultant acceleration
- `accel_std`: Standard deviation of resultant acceleration
- `accel_min`: Minimum resultant acceleration
- `accel_max`: Maximum resultant acceleration
- `accel_range`: Range (max - min) of resultant acceleration

### 2. Basic Statistics (Gyroscope)

- `gyro_mean`: Mean resultant angular velocity (if available)
- `gyro_std`: Standard deviation of resultant angular velocity (if available)
- `gyro_max`: Maximum resultant angular velocity (if available)

### 3. G-Seconds Load Metrics

- `time_above_2.0g`: Total time (seconds) above 2.0 g threshold
- `time_above_3.0g`: Total time (seconds) above 3.0 g threshold
- `g_seconds_2.0g`: Cumulative G-seconds above 2.0 g (integrated load)
- `g_seconds_3.0g`: Cumulative G-seconds above 3.0 g (integrated load)

### 4. Peak/Impact Features

- `num_peaks_over_3g`: Number of samples exceeding 3.0 g
- `num_peaks_over_4g`: Number of samples exceeding 4.0 g
- `highest_peak_g`: Highest detected peak above 2.0 g (using scipy peak detection)
  - If no peaks above 2g exist, returns 0
  - Uses `find_peaks` with height=2.0g and minimum distance of 10ms

### 5. Continuous High-G Duration

- `longest_2g_duration`: Longest continuous segment above 2.0 g (seconds)

### 6. Jerk (Rate of Change)

- `jerk_mean`: Mean absolute jerk value
- `jerk_max`: Maximum absolute jerk value

### 7. Frequency Features

- `dominant_freq`: Dominant frequency from FFT analysis (Hz)

### 8. Run Metadata

- `run_duration`: Total duration of the run (seconds)
- `athlete_id`: Athlete identifier
- `run_id`: Run identifier

## Usage

### Basic Usage

```python
from sledhead_imu.features import extract_all_runs
import pandas as pd

# Load raw IMU data
df = pd.read_csv('imu_data.csv')

# Extract features for all runs
features_df = extract_all_runs(df, fs=2000.0)

# Result: DataFrame with one row per run, 22 feature columns
print(features_df.head())
```

### Single Run Extraction

```python
from sledhead_imu.features import extract_rf_features

# Extract features for a single run
features = extract_rf_features(df_run, fs=2000.0)

# Result: Dictionary of feature values
print(features)
```

### Custom Column Names

```python
# If your data has different column names
accel_cols = {'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}
gyro_cols = {'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}

features = extract_rf_features(
    df_run,
    fs=2000.0,
    accel_cols=accel_cols,
    gyro_cols=gyro_cols
)
```

## Input Requirements

### Required Columns

- `athlete_id`: Unique athlete identifier
- `run_id`: Unique run identifier
- `timestamp`: Time series timestamp (used for duration calculation)
- Accelerometer data: Either `(g_x, g_y, g_z)` or `(x, y, z)` columns

### Optional Columns

- Gyroscope data: `(gx, gy, gz)` or `(gyrox, gyroy, gyroz)` columns
  - Note: Currently skipped if raw ADC values (needs conversion)

### Data Assumptions

- **Sampling rate**: 2000 Hz (configurable via `fs` parameter)
- **Acceleration units**: Already in g units (standard gravity: 9.80665 m/s²)
  - No conversion needed - values should already be normalized (e.g., 1.0 = 1g = 9.80665 m/s²)
  - If your data is in m/s², convert first: `accel_g = accel_mps2 / 9.80665`
- **Timestamp format**: Compatible with `pd.to_datetime()`

## Output Format

### DataFrame Structure

```
   athlete_id run_id  accel_mean  accel_std  ...  run_duration
0         A001   R001     1.00713    0.12345  ...         1.006
1         A002   R001     1.00376    0.15432  ...         0.556
```

### Data Types

- All numeric features: `float64`
- Identifier fields: `object` (string)
- Feature values: None are `NaN` or `np.nan`

## Integration with Machine Learning

### Scikit-learn Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Prepare data
X = features_df.drop(['athlete_id', 'run_id'], axis=1)
y = symptom_labels  # Your binary labels

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Predict
predictions = rf.predict(X)
```

### Feature Importance

```python
# Get feature importances
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importances)
```

## Implementation Details

### Resultant Acceleration

```python
accel_res = sqrt(g_x² + g_y² + g_z²)
```

### G-Seconds Calculation

```python
dt = 1.0 / fs
load_samples = max(accel_res - threshold, 0)
g_seconds = sum(load_samples * dt)
```

### Jerk Calculation

```python
jerk = diff(accel_res) / dt
jerk_mean = mean(abs(jerk))
jerk_max = max(abs(jerk))
```

### Dominant Frequency

```python
fft_vals = np.fft.rfft(accel_res)
fft_mag = np.abs(fft_vals)
freqs = np.fft.rfftfreq(len(accel_res), dt)
dominant_freq = freqs[argmax(fft_mag[1:]) + 1]  # Ignore DC component
```

## Testing

Run the test suite:

```bash
PYTHONPATH=src python -m pytest tests/test_features_random_forest.py -v
```

## File Location

- **Implementation**: `src/sledhead_imu/features/random_forest_features.py`
- **Tests**: `tests/test_features_random_forest.py`
- **Notebook**: `notebooks/06_features_random_forest.ipynb`

## References

- Features inspired by concussion research on head impact exposure metrics
- G-seconds concept based on cumulative load models for impact assessment
- Jerk metrics for rapid acceleration changes
- Frequency analysis for identifying dominant motion patterns
