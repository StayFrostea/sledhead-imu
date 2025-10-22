#!/bin/bash
# Validate model cutoffs

echo "Validating model cutoffs..."
python -m sledhead_imu.validate.validate_cutoffs
echo "Cutoff validation completed!"
