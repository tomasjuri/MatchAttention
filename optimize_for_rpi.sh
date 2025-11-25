#!/bin/bash
# =============================================================================
# MatchAttention Optimization Pipeline for Raspberry Pi 4
# =============================================================================
# This script automates the full optimization process:
# 1. Lists available presets and their configurations
# 2. Exports models to ONNX with different optimization levels
# 3. Applies INT8 quantization for smaller size and faster inference
# 4. Benchmarks all configurations
# 5. Generates a comparison report
#
# Usage:
#   ./optimize_for_rpi.sh [checkpoint_path] [output_dir]
#
# Example:
#   ./optimize_for_rpi.sh checkpoints/matchstereo_tiny_fsd.pth exports/
# =============================================================================

set -e  # Exit on error

# Default values
CHECKPOINT="${1:-checkpoints/matchstereo_tiny_fsd.pth}"
OUTPUT_DIR="${2:-exports}"
VARIANT="tiny"
HEIGHT=384
WIDTH=384

echo "============================================================================="
echo "MatchAttention Optimization Pipeline for Raspberry Pi 4"
echo "============================================================================="
echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output dir: $OUTPUT_DIR"
echo "  Variant: $VARIANT"
echo "  Resolution: ${HEIGHT}x${WIDTH}"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Please download the checkpoint first:"
    echo "  wget https://huggingface.co/Tingman/MatchAttention/resolve/main/matchstereo_tiny_fsd.pth"
    echo "  mv matchstereo_tiny_fsd.pth checkpoints/"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================================="
echo "Step 1: List Available Optimization Presets"
echo "============================================================================="
python -c "from models.fast_match_stereo import list_presets; list_presets()"

echo ""
echo "============================================================================="
echo "Step 2: Export All Presets to ONNX"
echo "============================================================================="
python export_onnx.py \
    --checkpoint_path "$CHECKPOINT" \
    --variant "$VARIANT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --export_all \
    --quantize int8 \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "============================================================================="
echo "Step 3: Benchmark PyTorch Models"
echo "============================================================================="
python benchmark_models.py \
    --checkpoint_path "$CHECKPOINT" \
    --variant "$VARIANT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --mode presets \
    --device_id -1 \
    --num_runs 5

echo ""
echo "============================================================================="
echo "Step 4: Benchmark ONNX Models"
echo "============================================================================="
python benchmark_models.py \
    --checkpoint_path "$CHECKPOINT" \
    --variant "$VARIANT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --mode comparison \
    --onnx_dir "$OUTPUT_DIR" \
    --device_id -1 \
    --num_runs 5

echo ""
echo "============================================================================="
echo "Step 5: Generate Full Benchmark Report"
echo "============================================================================="
python benchmark_models.py \
    --checkpoint_path "$CHECKPOINT" \
    --variant "$VARIANT" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --mode full \
    --onnx_dir "$OUTPUT_DIR" \
    --device_id -1 \
    --num_runs 5 \
    --output "$OUTPUT_DIR/benchmark_report.json"

echo ""
echo "============================================================================="
echo "OPTIMIZATION COMPLETE!"
echo "============================================================================="
echo ""
echo "Generated files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*.onnx 2>/dev/null || echo "  (no ONNX files yet)"
echo ""
echo "Benchmark report: $OUTPUT_DIR/benchmark_report.json"
echo ""
echo "Recommended for Raspberry Pi 4:"
echo "  - Use: ${OUTPUT_DIR}/matchstereo_tiny_rpi4_optimized_int8.onnx"
echo "  - Or:  ${OUTPUT_DIR}/matchstereo_tiny_faster_int8.onnx"
echo ""
echo "To run inference on RPi4:"
echo "  python run_img_optimized.py \\"
echo "    --preset rpi4_optimized \\"
echo "    --checkpoint_path $CHECKPOINT \\"
echo "    --img0_dir images/left/ \\"
echo "    --img1_dir images/right/ \\"
echo "    --device_id -1 \\"
echo "    --inference_size $HEIGHT $WIDTH"
echo ""

