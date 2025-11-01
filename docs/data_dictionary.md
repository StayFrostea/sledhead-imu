# Data Dictionary

## Core Fields

### Identifiers
- **`athlete_id`**: string/int - Unique identifier for each athlete
- **`run_id`**: string/int - Unique identifier for each run/session
- **`timestamp`**: UTC ISO8601 - Timestamp of the measurement

### IMU Data
- **`g_x`**: float - Acceleration in X-axis (g units)
- **`g_y`**: float - Acceleration in Y-axis (g units)  
- **`g_z`**: float - Acceleration in Z-axis (g units)
- **`g_mag`**: float - Magnitude of acceleration vector (g units)
- **Note**: Values are already normalized to standard gravity (1g = 9.80665 m/s²)

### Symptom Data
- **`symptom_flags`**: binary or categorical - Symptom indicators after merge
- **`symptom_type`**: string - Type of symptom (e.g., headache, dizziness)
- **`severity`**: int - Severity level (1-5 scale)
- **`duration_minutes`**: float - Duration of symptom in minutes

### Derived Features
- **`exposure_s`**: float - Total exposure time above threshold (seconds)
- **`duration_s`**: float - Duration above threshold (seconds)
- **`g_mag_mean`**: float - Mean acceleration magnitude
- **`g_mag_max`**: float - Maximum acceleration magnitude
- **`g_mag_std`**: float - Standard deviation of acceleration magnitude
- **`sample_count`**: int - Number of samples in aggregation period

### Model Features
- **`feature_*`**: float - Engineered features for modeling
- **`label`**: binary/int - Target variable for training
- **`prediction`**: float - Model prediction output
- **`alert_level`**: string - Alert level based on thresholds
