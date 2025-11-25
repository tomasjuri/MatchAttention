#!/usr/bin/env python3
"""
ONNX Export Script for MatchStereo with multiple optimization levels.

This script exports MatchStereo models to ONNX format with various optimization
presets, suitable for deployment on embedded devices like Raspberry Pi 4.

The export process:
1. Creates models with different optimization presets
2. Exports each to ONNX format
3. Optionally applies quantization (INT8, FP16)
4. Validates the exported models

Usage:
    # Export single preset
    python export_onnx.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --preset fast --output_dir exports/

    # Export all presets for comparison
    python export_onnx.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --export_all --output_dir exports/

    # Export with quantization
    python export_onnx.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --preset rpi4_optimized --quantize int8 --output_dir exports/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.fast_match_stereo import FastMatchStereo, OPTIMIZATION_PRESETS, list_presets


def create_dummy_inputs(height, width, device='cpu', dtype=torch.float32):
    """Create dummy input tensors for ONNX export."""
    img0 = torch.randint(0, 256, (1, 3, height, width), dtype=dtype, device=device)
    img1 = torch.randint(0, 256, (1, 3, height, width), dtype=dtype, device=device)
    return img0, img1


class MatchStereoWrapper(torch.nn.Module):
    """
    Wrapper class for ONNX export that simplifies the output structure.
    
    The original model returns a dict, but ONNX prefers tuple outputs.
    This wrapper extracts only the main disparity output.
    """
    
    def __init__(self, model, output_mode='disparity'):
        super().__init__()
        self.model = model
        self.output_mode = output_mode
    
    def forward(self, img0, img1):
        """
        Forward pass for ONNX export.
        
        Args:
            img0: Left image [B, 3, H, W] in range [0, 255]
            img1: Right image [B, 3, H, W] in range [0, 255]
            
        Returns:
            disparity: Disparity map [B, H, W] (positive values)
        """
        results = self.model(img0, img1, stereo=True)
        field_up = results['field_up']  # [B, H, W, 2]
        
        # Split bidirectional output and take left disparity
        B = field_up.shape[0] // 2
        disparity = -field_up[:B, :, :, 0]  # Negate for positive disparity
        disparity = torch.clamp(disparity, min=0)
        
        if self.output_mode == 'disparity':
            return disparity
        elif self.output_mode == 'disparity_with_confidence':
            # Also return the occlusion/confidence mask
            init_cv = results.get('init_cv', None)
            if init_cv is not None:
                confidence = init_cv[:B].max(dim=1)[0]  # Max correlation as confidence
                return disparity, confidence
            return disparity
        else:
            return disparity


def export_to_onnx(model, output_path, height, width, opset_version=17, 
                   dynamic_axes=True, simplify=True, verbose=False):
    """
    Export a MatchStereo model to ONNX format.
    
    Args:
        model: FastMatchStereo model
        output_path: Path to save ONNX model
        height: Input image height
        width: Input image width
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic input shapes
        simplify: Whether to simplify the ONNX model
        verbose: Print verbose export info
        
    Returns:
        bool: Success status
    """
    model.eval()
    
    # Create wrapper for cleaner ONNX output
    wrapped_model = MatchStereoWrapper(model, output_mode='disparity')
    wrapped_model.eval()
    
    # Create dummy inputs
    img0, img1 = create_dummy_inputs(height, width)
    
    # Define input/output names
    input_names = ['left_image', 'right_image']
    output_names = ['disparity']
    
    # Dynamic axes for variable input sizes
    if dynamic_axes:
        dynamic_axes_dict = {
            'left_image': {0: 'batch', 2: 'height', 3: 'width'},
            'right_image': {0: 'batch', 2: 'height', 3: 'width'},
            'disparity': {0: 'batch', 1: 'height', 2: 'width'},
        }
    else:
        dynamic_axes_dict = None
    
    print(f"  Exporting to ONNX (opset {opset_version})...")
    
    try:
        # Export to ONNX
        torch.onnx.export(
            wrapped_model,
            (img0, img1),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
            verbose=verbose,
        )
        
        print(f"  Exported to: {output_path}")
        
        # Simplify if requested and onnx-simplifier is available
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify
                
                print("  Simplifying ONNX model...")
                onnx_model = onnx.load(output_path)
                simplified_model, check = onnx_simplify(onnx_model)
                
                if check:
                    onnx.save(simplified_model, output_path)
                    print("  Model simplified successfully")
                else:
                    print("  Warning: Simplification check failed, keeping original")
                    
            except ImportError:
                print("  Note: Install onnxsim for model simplification: pip install onnxsim")
            except Exception as e:
                print(f"  Warning: Simplification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  Error exporting to ONNX: {e}")
        return False


def quantize_onnx(input_path, output_path, quantize_type='int8'):
    """
    Quantize an ONNX model.
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
        quantize_type: 'int8' or 'fp16'
        
    Returns:
        bool: Success status
    """
    try:
        if quantize_type == 'int8':
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            print(f"  Applying INT8 dynamic quantization...")
            quantize_dynamic(
                input_path,
                output_path,
                weight_type=QuantType.QUInt8,
            )
            print(f"  Quantized model saved to: {output_path}")
            return True
            
        elif quantize_type == 'fp16':
            import onnx
            from onnxconverter_common import float16
            
            print(f"  Converting to FP16...")
            model = onnx.load(input_path)
            model_fp16 = float16.convert_float_to_float16(model)
            onnx.save(model_fp16, output_path)
            print(f"  FP16 model saved to: {output_path}")
            return True
            
        else:
            print(f"  Unknown quantization type: {quantize_type}")
            return False
            
    except ImportError as e:
        print(f"  Error: Missing dependency for quantization: {e}")
        print(f"  Install with: pip install onnxruntime onnxconverter-common")
        return False
    except Exception as e:
        print(f"  Error during quantization: {e}")
        return False


def validate_onnx(onnx_path, height, width):
    """
    Validate an ONNX model by running inference.
    
    Args:
        onnx_path: Path to ONNX model
        height: Test input height
        width: Test input width
        
    Returns:
        tuple: (success, inference_time_ms)
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Check model is valid
        print("  Validating ONNX model...")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        
        # Create inference session
        session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider']
        )
        
        # Create test inputs
        img0 = np.random.randint(0, 256, (1, 3, height, width)).astype(np.float32)
        img1 = np.random.randint(0, 256, (1, 3, height, width)).astype(np.float32)
        
        # Warmup
        _ = session.run(None, {'left_image': img0, 'right_image': img1})
        
        # Benchmark
        times = []
        for _ in range(3):
            start = time.perf_counter()
            outputs = session.run(None, {'left_image': img0, 'right_image': img1})
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        disparity = outputs[0]
        
        print(f"  Validation passed!")
        print(f"    Output shape: {disparity.shape}")
        print(f"    Output range: [{disparity.min():.2f}, {disparity.max():.2f}]")
        print(f"    Inference time (CPU): {avg_time:.2f} ms")
        
        return True, avg_time
        
    except ImportError:
        print("  Warning: onnxruntime not installed, skipping validation")
        return True, None
    except Exception as e:
        print(f"  Validation failed: {e}")
        return False, None


def get_model_size(path):
    """Get model file size in MB."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0


def export_all_presets(args):
    """Export models for all optimization presets."""
    print("\n" + "=" * 80)
    print("EXPORTING ALL OPTIMIZATION PRESETS TO ONNX")
    print("=" * 80)
    
    results = []
    
    for preset_name in OPTIMIZATION_PRESETS.keys():
        print(f"\n--- Preset: {preset_name} ---")
        preset_config = OPTIMIZATION_PRESETS[preset_name]
        print(f"  Description: {preset_config['description']}")
        print(f"  Iterations: {sum(preset_config['refine_nums'])}")
        
        # Create model
        model = FastMatchStereo(args, preset=preset_name)
        
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        
        model.eval()
        
        # Export paths
        base_name = f"matchstereo_{args.variant}_{preset_name}"
        onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
        
        # Export to ONNX
        success = export_to_onnx(
            model, onnx_path, args.height, args.width,
            opset_version=args.opset_version,
            dynamic_axes=not args.fixed_size,
            simplify=args.simplify,
        )
        
        if not success:
            results.append({
                'preset': preset_name,
                'success': False,
            })
            continue
        
        result = {
            'preset': preset_name,
            'success': True,
            'onnx_path': onnx_path,
            'onnx_size_mb': get_model_size(onnx_path),
        }
        
        # Export both FP16 and INT8 versions
        fp16_path = os.path.join(args.output_dir, f"{base_name}_fp16.onnx")
        int8_path = os.path.join(args.output_dir, f"{base_name}_int8.onnx")
        
        fp16_success = quantize_onnx(onnx_path, fp16_path, 'fp16')
        int8_success = quantize_onnx(onnx_path, int8_path, 'int8')
        
        if fp16_success:
            result['fp16_path'] = fp16_path
            result['fp16_size_mb'] = get_model_size(fp16_path)
        
        if int8_success:
            result['int8_path'] = int8_path
            result['int8_size_mb'] = get_model_size(int8_path)
        
        # Validate
        if args.validate:
            valid, infer_time = validate_onnx(onnx_path, args.height, args.width)
            result['valid'] = valid
            result['inference_time_ms'] = infer_time
            
            if fp16_success:
                valid_fp16, infer_time_fp16 = validate_onnx(fp16_path, args.height, args.width)
                result['valid_fp16'] = valid_fp16
                result['inference_time_fp16_ms'] = infer_time_fp16
            
            if int8_success:
                valid_int8, infer_time_int8 = validate_onnx(int8_path, args.height, args.width)
                result['valid_int8'] = valid_int8
                result['inference_time_int8_ms'] = infer_time_int8
        
        results.append(result)
        
        # Clean up
        del model
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)
    print(f"\n{'Preset':<15} {'Success':<8} {'FP32 (MB)':<12} {'FP16 (MB)':<12} {'INT8 (MB)':<12} {'Infer (ms)':<12}")
    print("-" * 80)
    
    for r in results:
        success = "✓" if r.get('success') else "✗"
        fp32_size = f"{r.get('onnx_size_mb', 0):.2f}" if r.get('onnx_size_mb') else "-"
        fp16_size = f"{r.get('fp16_size_mb', 0):.2f}" if r.get('fp16_size_mb') else "-"
        int8_size = f"{r.get('int8_size_mb', 0):.2f}" if r.get('int8_size_mb') else "-"
        infer = f"{r.get('inference_time_ms', 0):.2f}" if r.get('inference_time_ms') else "-"
        print(f"{r['preset']:<15} {success:<8} {fp32_size:<12} {fp16_size:<12} {int8_size:<12} {infer:<12}")
    
    print("=" * 80)
    print(f"\nExported models saved to: {args.output_dir}")
    print("\nAll presets exported with FP32, FP16, and INT8 versions.")
    
    return results


def export_single_preset(args):
    """Export a single preset to ONNX."""
    print(f"\nExporting preset: {args.preset}")
    
    preset_config = OPTIMIZATION_PRESETS[args.preset]
    print(f"  Description: {preset_config['description']}")
    print(f"  Iterations: {sum(preset_config['refine_nums'])}")
    
    # Create model
    model = FastMatchStereo(args, preset=args.preset)
    
    if args.checkpoint_path:
        print(f"  Loading checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        if missing:
            print(f"  Note: {len(missing)} missing keys (expected for optimized configs)")
    
    model.eval()
    
    # Export paths
    base_name = f"matchstereo_{args.variant}_{args.preset}"
    onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
    
    # Export to ONNX
    success = export_to_onnx(
        model, onnx_path, args.height, args.width,
        opset_version=args.opset_version,
        dynamic_axes=not args.fixed_size,
        simplify=args.simplify,
        verbose=args.verbose,
    )
    
    if not success:
        print("Export failed!")
        return None
    
    print(f"  Model size: {get_model_size(onnx_path):.2f} MB")
    
    # Export both FP16 and INT8 versions
    quantized_paths = {}
    
    # Export FP16
    print("\n  Exporting FP16 version...")
    fp16_path = os.path.join(args.output_dir, f"{base_name}_fp16.onnx")
    if quantize_onnx(onnx_path, fp16_path, 'fp16'):
        quantized_paths['fp16'] = fp16_path
        print(f"  FP16 model size: {get_model_size(fp16_path):.2f} MB")
    
    # Export INT8
    print("\n  Exporting INT8 version...")
    int8_path = os.path.join(args.output_dir, f"{base_name}_int8.onnx")
    if quantize_onnx(onnx_path, int8_path, 'int8'):
        quantized_paths['int8'] = int8_path
        print(f"  INT8 model size: {get_model_size(int8_path):.2f} MB")
    
    # Validate all exported models
    if args.validate:
        print("\n  Validating models...")
        print("\n  Validating FP32 model:")
        validate_onnx(onnx_path, args.height, args.width)
        
        if 'fp16' in quantized_paths:
            print("\n  Validating FP16 model:")
            validate_onnx(quantized_paths['fp16'], args.height, args.width)
        
        if 'int8' in quantized_paths:
            print("\n  Validating INT8 model:")
            validate_onnx(quantized_paths['int8'], args.height, args.width)
    
    print("\nExport complete!")
    print(f"  Exported models:")
    print(f"    - FP32: {onnx_path} ({get_model_size(onnx_path):.2f} MB)")
    if 'fp16' in quantized_paths:
        print(f"    - FP16: {quantized_paths['fp16']} ({get_model_size(quantized_paths['fp16']):.2f} MB)")
    if 'int8' in quantized_paths:
        print(f"    - INT8: {quantized_paths['int8']} ({get_model_size(quantized_paths['int8']):.2f} MB)")
    
    return onnx_path


def main():
    parser = argparse.ArgumentParser(
        description="Export MatchStereo models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single preset
  python export_onnx.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \\
      --preset fast --output_dir exports/

  # Export all presets (automatically exports FP32, FP16, and INT8)
  python export_onnx.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \\
      --export_all --output_dir exports/

  # Export for fixed 400x400 input size
  python export_onnx.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \\
      --preset rpi4_optimized --height 384 --width 384 --fixed_size \\
      --output_dir exports/
"""
    )
    
    # Model configuration
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--preset', type=str, default='rpi4_optimized',
                        choices=list(OPTIMIZATION_PRESETS.keys()),
                        help='Optimization preset to export')
    parser.add_argument('--variant', choices=['tiny', 'small', 'base'], default='tiny',
                        help='Model variant')
    parser.add_argument('--mat_impl', choices=['pytorch', 'cuda'], default='pytorch',
                        help='MatchAttention implementation (must be pytorch for ONNX)')
    
    # Export settings
    parser.add_argument('--output_dir', type=str, default='exports',
                        help='Output directory for ONNX models')
    parser.add_argument('--height', type=int, default=384,
                        help='Export input height (should be divisible by 32)')
    parser.add_argument('--width', type=int, default=384,
                        help='Export input width (should be divisible by 32)')
    parser.add_argument('--opset_version', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--fixed_size', action='store_true',
                        help='Export with fixed input size (no dynamic axes)')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Simplify ONNX model (requires onnxsim)')
    parser.add_argument('--no_simplify', dest='simplify', action='store_false')
    
    # Quantization (deprecated - now exports both FP16 and INT8 by default)
    parser.add_argument('--quantize', type=str, choices=['int8', 'fp16'],
                        help='[DEPRECATED] Both FP16 and INT8 are now exported automatically')
    
    # Validation
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate exported model')
    parser.add_argument('--no_validate', dest='validate', action='store_false')
    
    # Batch export
    parser.add_argument('--export_all', action='store_true',
                        help='Export all optimization presets')
    
    # Misc
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose export output')
    parser.add_argument('--list_presets', action='store_true',
                        help='List available presets and exit')

    args = parser.parse_args()
    
    if args.list_presets:
        list_presets()
        return
    
    # Validate input size
    if args.height % 32 != 0 or args.width % 32 != 0:
        print(f"Warning: Input size ({args.height}x{args.width}) should be divisible by 32")
        args.height = (args.height // 32) * 32
        args.width = (args.width // 32) * 32
        print(f"  Adjusted to: {args.height}x{args.width}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Force PyTorch implementation for ONNX export
    if args.mat_impl != 'pytorch':
        print("Note: Forcing mat_impl=pytorch for ONNX export")
        args.mat_impl = 'pytorch'
    
    # Export
    if args.export_all:
        export_all_presets(args)
    else:
        export_single_preset(args)


if __name__ == "__main__":
    main()

