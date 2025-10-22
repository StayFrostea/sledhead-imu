# Data Directory Structure

This directory contains the Sled-Head IMU pipeline data organized by processing stage.

## Directory Structure

- `00_collect/` - Raw input data
  - `imu/` - IMU sensor data files
  - `symptoms/` - Symptom data files  
  - `metadata/` - Athlete and run metadata
- `01_ingest_normalize/` - Cleaned, normalized data
- `02_detect_trim_runs/` - Trimmed run segments
- `03_segment_by_athlete/` - Athlete-specific segments
- `04_aggregate_per_day/` - Daily aggregated data
- `05_filtering/` - Filtered data (HP, LP, rolling avg)
- `06_features_exposure/` - ≥2g exposure metrics
- `07_labels_merge/` - Merged symptom labels
- `08_model_ready/` - Final feature tables
- `09_splits/` - Train/test/validation splits
- `10_models/` - Trained models
- `11_metrics_validate_cutoffs/` - Validation results and thresholds
- `12_predictions_alerts/` - Final predictions and alerts

## Sample Data

Sample files are provided for testing and development:
- `sample_imu_*.csv` - Sample IMU data files
- `sample_symptoms_data.csv` - Sample symptom data

## Data Formats

- **IMU Data**: CSV files with columns: timestamp, athlete_id, run_id, accy, accz, gyrox, gyroy, gyroz, t, x, y, z, r_gs, num_symptoms
- **Symptom Data**: CSV files with columns: timestamp, athlete_id, symptom_type, severity, duration_minutes

## Large Files

Large data files should be stored using Git LFS or external storage (S3, etc.) and not committed directly to the repository.
