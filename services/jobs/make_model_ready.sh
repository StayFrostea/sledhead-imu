#!/bin/bash
# Make model-ready data

echo "Building model-ready dataset..."
python -m sledhead_imu.prep.build_model_ready
echo "Model-ready data created successfully!"
