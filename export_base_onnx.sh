#!/bin/bash

# Export MatchStereo Base model to ONNX format (FP32, FP16, INT8)
# This script exports all three precision versions automatically

CHECKPOINT_PATH="/Users/tomasjurica/projects/FingernailCalibration/StereoVision/MatchAttention/checkpoints/matchstereo_tiny_fsd.pth"
OUTPUT_DIR="exports"
VARIANT="tiny"
# Use 416x416 (divisible by 32) instead of 400x400
HEIGHT=416
WIDTH=416

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please download the base model checkpoint first"
    echo "Please ensure the checkpoint file exists at the specified path"
    exit 1
fi

echo "=========================================="
echo "Exporting MatchStereo Tiny to ONNX"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Variant: $VARIANT"
echo "Input size: ${HEIGHT}x${WIDTH}"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "This will export FP32, FP16, and INT8 versions automatically"
echo ""

# Export - the script automatically creates FP32, FP16, and INT8 versions
# Use fixed_size to avoid dynamic shape issues
# python3 export_onnx.py \
#     --checkpoint_path "$CHECKPOINT_PATH" \
#     --variant "$VARIANT" \
#     --preset fast \
#     --height "$HEIGHT" \
#     --width "$WIDTH" \
#     --output_dir "$OUTPUT_DIR" \
#     --opset_version 18 \
#     --fixed_size \
#     --simplify \
#     --validate

echo ""
echo "=========================================="
echo "Exporting all presets for comparison"
echo "=========================================="

# Export all presets for comparison
python3 export_onnx.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --variant "$VARIANT" \
    --export_all \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --output_dir "$OUTPUT_DIR" \
    --opset_version 18 \
    --fixed_size \
    --simplify \
    --validate

echo ""
echo "=========================================="
echo "Exporting RPI4 optimized preset with INT8"
echo "=========================================="

# Export with quantization (RPI4 optimized preset)
python3 export_onnx.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --variant "$VARIANT" \
    --preset rpi4_optimized \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --output_dir "$OUTPUT_DIR" \
    --opset_version 18 \
    --fixed_size \
    --simplify \
    --validate

echo ""
echo "=========================================="
echo "Export complete!"
echo "=========================================="
echo ""
echo "Exported models in $OUTPUT_DIR:"
echo "  - matchstereo_tiny_fast.onnx (FP32)"
echo "  - matchstereo_tiny_fast_fp16.onnx (FP16)"
echo "  - matchstereo_tiny_fast_int8.onnx (INT8)"

