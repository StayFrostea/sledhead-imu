# ≥2 g Exposure Metric (Draft)

## Overview
This document defines the ≥2 g exposure metric used in the Sled-Head IMU pipeline for quantifying head impact exposure.

## Input Requirements
- **Data Source**: Filtered acceleration magnitude or axis of interest (post HP/LP + rolling average)
- **Required Columns**: 
  - `athlete_id`: Unique athlete identifier
  - `run_id`: Unique run identifier  
  - `timestamp`: Time series timestamp
  - `g_mag` or specified axis: Acceleration magnitude or specific axis values

## Calculation Method
1. **Threshold Application**: Count only samples with g >= 2.0
2. **Dose Calculation**: Integrate g * Δt to get a "dose"-like value per run/day
3. **Aggregation**: Sum exposure across time periods

## Output Tables
- **Per-run exposure**: `exposure_s` (total exposure in seconds), `duration_s` (time above threshold)
- **Per-day exposure**: Daily aggregated exposure metrics

## Validation Notes
- To be versioned as we validate cutoffs
- Threshold of 2.0g is initial estimate, subject to validation
- Integration method may be refined based on clinical validation

## Implementation
See `src/sledhead_imu/features/exposure_2g.py` for the current implementation.
