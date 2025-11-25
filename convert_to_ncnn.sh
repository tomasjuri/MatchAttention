#!/bin/bash

# Convert ONNX model to NCNN format
# Requires: pip install pnnx

ONNX_MODEL="/home/tomasjurica/projects/MatchAttention/exports/matchstereo_tiny_default.onnx"

echo "Converting ONNX model to NCNN format..."
echo "  Input: $ONNX_MODEL"

python3 export_ncnn.py \
    --onnx_path "$ONNX_MODEL"

echo ""
echo "If conversion was successful, run inference with:"
echo "  ./run_ncnn.sh"

