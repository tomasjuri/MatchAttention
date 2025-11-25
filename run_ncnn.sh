#!/bin/bash

# Simple shell script to run MatchStereo inference with NCNN models
# Requires: pip install ncnn
# Optional: Vulkan SDK for GPU acceleration

IMG0_DIR="/home/tomasjurica/projects/MatchAttention/input_data_416/left"
IMG1_DIR="/home/tomasjurica/projects/MatchAttention/input_data_416/right"
NCNN_MODEL="/home/tomasjurica/projects/MatchAttention/exports/matchstereo_tiny_default.ncnn.param"
OUTPUT_PATH="outputs_ncnn"

# Check if NCNN model exists
if [ ! -f "$NCNN_MODEL" ]; then
    echo "NCNN model not found: $NCNN_MODEL"
    echo "Run ./convert_to_ncnn.sh first to convert the ONNX model"
    exit 1
fi

python3 run_img.py \
    --img0_dir "$IMG0_DIR" \
    --img1_dir "$IMG1_DIR" \
    --checkpoint_path "$NCNN_MODEL" \
    --output_path "$OUTPUT_PATH" \
    --save_right \
    --device_id -1

