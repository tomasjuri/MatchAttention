#!/usr/bin/env python3
"""
Convert ONNX model to NCNN format.

This script converts an ONNX model to NCNN's .param and .bin files using pnnx.

Usage:
    python export_ncnn.py --onnx_path exports/matchstereo_tiny_default.onnx

Requirements:
    pip install pnnx
"""

import argparse
import os
import subprocess
import sys


def convert_onnx_to_ncnn(onnx_path, output_dir=None):
    """Convert ONNX model to NCNN format using pnnx."""
    
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(onnx_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name without extension
    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    
    print(f"Converting ONNX model to NCNN format...")
    print(f"  Input: {onnx_path}")
    print(f"  Output directory: {output_dir}")
    
    # Try using pnnx first (recommended)
    try:
        import pnnx
        print("  Using pnnx for conversion...")
        
        # pnnx must run from the ONNX file's directory to find external data files (.onnx.data)
        onnx_dir = os.path.dirname(os.path.abspath(onnx_path))
        onnx_filename = os.path.basename(onnx_path)
        original_dir = os.getcwd()
        os.chdir(onnx_dir)
        
        try:
            # pnnx converts ONNX to NCNN format
            result = subprocess.run(
                ['pnnx', onnx_filename],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"  pnnx stderr: {result.stderr}")
                print(f"  pnnx stdout: {result.stdout}")
                raise RuntimeError(f"pnnx failed with return code {result.returncode}")
            
            print(f"  pnnx stdout: {result.stdout}")
            
        finally:
            os.chdir(original_dir)
        
        # pnnx outputs to the same directory as the ONNX file
        param_file = os.path.join(onnx_dir, f"{base_name}.ncnn.param")
        bin_file = os.path.join(onnx_dir, f"{base_name}.ncnn.bin")
        
        # If output_dir is different, move files there
        if output_dir != onnx_dir:
            import shutil
            final_param = os.path.join(output_dir, f"{base_name}.ncnn.param")
            final_bin = os.path.join(output_dir, f"{base_name}.ncnn.bin")
            if os.path.exists(param_file):
                shutil.move(param_file, final_param)
                param_file = final_param
            if os.path.exists(bin_file):
                shutil.move(bin_file, final_bin)
                bin_file = final_bin
        
        if os.path.exists(param_file) and os.path.exists(bin_file):
            print(f"\nConversion successful!")
            print(f"  Param file: {param_file}")
            print(f"  Bin file: {bin_file}")
            return param_file, bin_file
        else:
            # List files in onnx_dir to see what was created
            print(f"\nChecking ONNX directory: {onnx_dir}")
            for f in os.listdir(onnx_dir):
                if base_name in f:
                    print(f"  Found: {f}")
            
            raise FileNotFoundError("NCNN output files not found after conversion")
            
    except ImportError:
        print("  pnnx not found, trying onnx2ncnn...")
        
        # Try onnx2ncnn tool (older method)
        param_file = os.path.join(output_dir, f"{base_name}.param")
        bin_file = os.path.join(output_dir, f"{base_name}.bin")
        
        result = subprocess.run(
            ['onnx2ncnn', onnx_path, param_file, bin_file],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"  onnx2ncnn stderr: {result.stderr}")
            raise RuntimeError(
                f"onnx2ncnn failed. Please install pnnx: pip install pnnx\n"
                f"Or build ncnn with onnx2ncnn tool from source."
            )
        
        print(f"\nConversion successful!")
        print(f"  Param file: {param_file}")
        print(f"  Bin file: {bin_file}")
        return param_file, bin_file


def internalize_external_data(onnx_path):
    """
    Convert ONNX model with external data to a single file with embedded weights.
    This is required for pnnx which doesn't handle external data files well.
    """
    try:
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data
        
        print(f"Internalizing external data...")
        
        # Check if external data file exists
        data_file = onnx_path + ".data"
        if not os.path.exists(data_file):
            print(f"  No external data file found, skipping")
            return onnx_path
        
        # Load model with external data
        model = onnx.load(onnx_path, load_external_data=True)
        
        # Save as single file (internalized)
        internalized_path = onnx_path.replace('.onnx', '_internalized.onnx')
        onnx.save(model, internalized_path)
        
        print(f"  Internalized model saved to: {internalized_path}")
        return internalized_path
        
    except Exception as e:
        print(f"  Failed to internalize: {e}")
        return onnx_path


def simplify_onnx_if_needed(onnx_path):
    """Optionally simplify ONNX model before conversion."""
    try:
        import onnx
        from onnxsim import simplify
        
        print(f"Simplifying ONNX model...")
        model = onnx.load(onnx_path, load_external_data=True)
        model_simp, check = simplify(model)
        
        if check:
            simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(model_simp, simplified_path)
            print(f"  Simplified model saved to: {simplified_path}")
            return simplified_path
        else:
            print("  Simplification failed, using original model")
            return onnx_path
            
    except ImportError:
        print("  onnx-simplifier not installed, skipping simplification")
        return onnx_path


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to NCNN format")
    parser.add_argument('--onnx_path', type=str, required=True,
                        help='Path to ONNX model file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for NCNN files (default: same as ONNX file)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model before conversion')
    parser.add_argument('--no_internalize', action='store_true',
                        help='Skip internalizing external data (not recommended)')
    
    args = parser.parse_args()
    
    onnx_path = args.onnx_path
    
    # Internalize external data (required for pnnx to work with .onnx.data files)
    if not args.no_internalize:
        onnx_path = internalize_external_data(onnx_path)
    
    # Optionally simplify ONNX model
    if args.simplify:
        onnx_path = simplify_onnx_if_needed(onnx_path)
    
    # Convert to NCNN
    try:
        param_file, bin_file = convert_onnx_to_ncnn(onnx_path, args.output_dir)
        print(f"\nTo use the NCNN model in run_img.py:")
        print(f"  python run_img.py --checkpoint_path {param_file} ...")
    except Exception as e:
        print(f"\nConversion failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Install pnnx: pip install pnnx")
        print("  2. Or build ncnn from source with onnx2ncnn tool")
        print("  3. Try simplifying the ONNX model: --simplify")
        sys.exit(1)


if __name__ == "__main__":
    main()

