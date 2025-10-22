from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
COLLECT = DATA / "00_collect"
INGEST = DATA / "01_ingest_normalize"
TRIM = DATA / "02_detect_trim_runs" / "segments"
DAILY = DATA / "04_aggregate_per_day" / "daily_athlete_data"
FILTERED = DATA / "05_filtering" / "filtered_data"
EXPOSURE = DATA / "06_features_exposure" / "datasets_daily_exposure"
LABELS = DATA / "07_labels_merge"
MODEL_READY = DATA / "08_model_ready"
SPLITS = DATA / "09_splits"
MODELS = DATA / "10_models"
METRICS = DATA / "11_metrics_validate_cutoffs"
PRED_OUT = DATA / "12_predictions_alerts" / "outputs"

os.makedirs(MODEL_READY, exist_ok=True)
