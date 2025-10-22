# Sled-Head IMU

End-to-end pipeline for IMU data: collect → ingest/normalize → run trim/segmentation → filtering (HP, LP, rolling avg) → ≥2g exposure metrics → label merge → model-ready → train/test (NN, MLR) → validate/define bench cutoffs → predictions + thresholds → bench/alert.

## Quickstart
```bash
make setup
source .venv/bin/activate
jupyter lab
```

## Data Layout

- `data/00_collect/imu`, `data/00_collect/symptoms` (raw inputs)
- Subsequent folders mirror each stage; each step writes to the next stage.

## Pipeline Stages

1. **Collect Data** → Raw IMU and symptom data
2. **Ingest & Normalize** → Cleaned, unified schema
3. **Detect & Trim Runs** → Trimmed run windows
4. **Segment by Athlete** → Athlete-specific segments
5. **Aggregate per Day** → Daily athlete data
6. **Filtering** → High pass, low pass, rolling average filters
7. **Compute ≥2 g Exposure Metrics** → Exposure dose calculations
8. **Attach Labels** → Merge symptom data
9. **Model-Ready Data** → Final feature table (X) + labels (y)
10. **Train/Test Split** → Training, validation, test sets
11. **Models** → Neural Network and MLR models
12. **Validate & Define Bench Cutoffs** → Model validation and thresholds
13. **Predictions + Bench Thresholds** → Final predictions and alerts
