#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for MatchStereo Models.

This script benchmarks MatchStereo models across:
- Different optimization presets
- PyTorch vs ONNX runtime
- Different input resolutions
- CPU vs GPU (if available)
- FP32 vs FP16 precision

The results help you choose the best configuration for your target device.

Usage:
    # Quick benchmark of presets
    python benchmark_models.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --mode presets

    # Compare PyTorch vs ONNX
    python benchmark_models.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --onnx_dir exports/ --mode comparison

    # Full benchmark (all modes)
    python benchmark_models.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --onnx_dir exports/ --mode full

    # Benchmark for specific resolution (e.g., 400x400 for RPi4)
    python benchmark_models.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --height 384 --width 384 --mode presets
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import psutil

from models.fast_match_stereo import FastMatchStereo, OPTIMIZATION_PRESETS


def get_system_info():
    """Gather system information for benchmark context."""
    info = {
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version.split()[0],
        'pytorch_version': torch.__version__,
        'cpu': {
            'count': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
        },
        'memory_gb': psutil.virtual_memory().total / (1024**3),
    }
    
    if torch.cuda.is_available():
        info['cuda'] = {
            'available': True,
            'device': torch.cuda.get_device_name(0),
            'version': torch.version.cuda,
        }
    else:
        info['cuda'] = {'available': False}
    
    # Check for ONNX Runtime
    try:
        import onnxruntime as ort
        info['onnxruntime_version'] = ort.__version__
        info['onnxruntime_providers'] = ort.get_available_providers()
    except ImportError:
        info['onnxruntime_version'] = None
    
    return info


def benchmark_pytorch_model(model, height, width, device, dtype, num_warmup=3, num_runs=10):
    """
    Benchmark a PyTorch model.
    
    Returns:
        dict with timing statistics
    """
    model.to(device).to(dtype).eval()
    
    # Create dummy inputs
    img0 = torch.randint(0, 256, (1, 3, height, width), dtype=dtype, device=device)
    img1 = torch.randint(0, 256, (1, 3, height, width), dtype=dtype, device=device)
    
    # Warmup
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(img0, img1, stereo=True)
    
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    memory_used = []
    
    with torch.inference_mode():
        for _ in range(num_runs):
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            start = time.perf_counter()
            output = model(img0, img1, stereo=True)
            
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
            
            if device != 'cpu' and torch.cuda.is_available():
                memory_used.append(torch.cuda.max_memory_allocated() / (1024**2))  # MB
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'memory_mb': np.mean(memory_used) if memory_used else None,
    }


def benchmark_onnx_model(onnx_path, height, width, num_warmup=3, num_runs=10):
    """
    Benchmark an ONNX model.
    
    Returns:
        dict with timing statistics
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ONNX Runtime not installed, skipping ONNX benchmark")
        return None
    
    # Create session
    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    
    # Create dummy inputs
    img0 = np.random.randint(0, 256, (1, 3, height, width)).astype(np.float32)
    img1 = np.random.randint(0, 256, (1, 3, height, width)).astype(np.float32)
    
    input_feed = {'left_image': img0, 'right_image': img1}
    
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, input_feed)
    
    # Benchmark
    times = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        outputs = session.run(None, input_feed)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'median_ms': np.median(times),
        'output_shape': outputs[0].shape,
    }


def benchmark_presets(args):
    """Benchmark all optimization presets."""
    print("\n" + "=" * 80)
    print("BENCHMARKING OPTIMIZATION PRESETS")
    print("=" * 80)
    
    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() and args.device_id >= 0 else 'cpu'
    dtype = torch.float32 if args.precision == 'fp32' else torch.float16
    
    print(f"\nConfiguration:")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Device: {device}")
    print(f"  Precision: {args.precision}")
    print(f"  Runs: {args.num_runs}")
    
    results = []
    
    for preset_name, preset_config in OPTIMIZATION_PRESETS.items():
        print(f"\n--- Testing: {preset_name} ---")
        print(f"  {preset_config['description']}")
        
        # Create model
        model = FastMatchStereo(args, preset=preset_name)
        
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        
        config = model.get_config_summary()
        
        # Benchmark
        stats = benchmark_pytorch_model(
            model, args.height, args.width, device, dtype,
            num_warmup=args.num_warmup, num_runs=args.num_runs
        )
        
        result = {
            'preset': preset_name,
            'iterations': config['total_iterations'],
            **stats,
        }
        results.append(result)
        
        print(f"  Iterations: {config['total_iterations']}")
        print(f"  Time: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        if stats.get('memory_mb'):
            print(f"  Memory: {stats['memory_mb']:.2f} MB")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary table
    print("\n" + "=" * 80)
    print("PRESET BENCHMARK SUMMARY")
    print("=" * 80)
    
    baseline_time = results[0]['mean_ms']
    
    print(f"\n{'Preset':<15} {'Iters':<8} {'Mean (ms)':<12} {'Std':<8} {'Speedup':<10}")
    print("-" * 60)
    
    for r in results:
        speedup = baseline_time / r['mean_ms']
        print(f"{r['preset']:<15} {r['iterations']:<8} {r['mean_ms']:.2f}        {r['std_ms']:.2f}    {speedup:.2f}x")
    
    return results


def benchmark_pytorch_vs_onnx(args):
    """Compare PyTorch models with ONNX exports."""
    print("\n" + "=" * 80)
    print("PYTORCH vs ONNX COMPARISON")
    print("=" * 80)
    
    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() and args.device_id >= 0 else 'cpu'
    dtype = torch.float32 if args.precision == 'fp32' else torch.float16
    
    print(f"\nConfiguration:")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  Device: {device}")
    print(f"  ONNX dir: {args.onnx_dir}")
    
    results = []
    
    presets_to_test = ['default', 'fast', 'faster', 'rpi4_optimized']
    
    for preset_name in presets_to_test:
        if preset_name not in OPTIMIZATION_PRESETS:
            continue
            
        print(f"\n--- Preset: {preset_name} ---")
        
        result = {'preset': preset_name}
        
        # PyTorch benchmark
        print("  PyTorch:")
        model = FastMatchStereo(args, preset=preset_name)
        
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        
        pytorch_stats = benchmark_pytorch_model(
            model, args.height, args.width, device, dtype,
            num_warmup=args.num_warmup, num_runs=args.num_runs
        )
        result['pytorch'] = pytorch_stats
        print(f"    Time: {pytorch_stats['mean_ms']:.2f} ± {pytorch_stats['std_ms']:.2f} ms")
        
        del model
        
        # ONNX benchmark
        onnx_path = os.path.join(args.onnx_dir, f"matchstereo_{args.variant}_{preset_name}.onnx")
        
        if os.path.exists(onnx_path):
            print("  ONNX (FP32):")
            onnx_stats = benchmark_onnx_model(
                onnx_path, args.height, args.width,
                num_warmup=args.num_warmup, num_runs=args.num_runs
            )
            if onnx_stats:
                result['onnx_fp32'] = onnx_stats
                print(f"    Time: {onnx_stats['mean_ms']:.2f} ± {onnx_stats['std_ms']:.2f} ms")
        
        # INT8 ONNX benchmark
        int8_path = os.path.join(args.onnx_dir, f"matchstereo_{args.variant}_{preset_name}_int8.onnx")
        
        if os.path.exists(int8_path):
            print("  ONNX (INT8):")
            int8_stats = benchmark_onnx_model(
                int8_path, args.height, args.width,
                num_warmup=args.num_warmup, num_runs=args.num_runs
            )
            if int8_stats:
                result['onnx_int8'] = int8_stats
                print(f"    Time: {int8_stats['mean_ms']:.2f} ± {int8_stats['std_ms']:.2f} ms")
        
        results.append(result)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Preset':<15} {'PyTorch (ms)':<15} {'ONNX FP32 (ms)':<15} {'ONNX INT8 (ms)':<15}")
    print("-" * 70)
    
    for r in results:
        pytorch = f"{r.get('pytorch', {}).get('mean_ms', 0):.2f}" if 'pytorch' in r else "-"
        onnx_fp32 = f"{r.get('onnx_fp32', {}).get('mean_ms', 0):.2f}" if 'onnx_fp32' in r else "-"
        onnx_int8 = f"{r.get('onnx_int8', {}).get('mean_ms', 0):.2f}" if 'onnx_int8' in r else "-"
        print(f"{r['preset']:<15} {pytorch:<15} {onnx_fp32:<15} {onnx_int8:<15}")
    
    return results


def benchmark_resolutions(args):
    """Benchmark across different input resolutions."""
    print("\n" + "=" * 80)
    print("RESOLUTION SCALING BENCHMARK")
    print("=" * 80)
    
    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() and args.device_id >= 0 else 'cpu'
    dtype = torch.float32
    
    resolutions = [
        (256, 256),
        (384, 384),
        (384, 512),
        (512, 512),
        (512, 640),
        (640, 640),
    ]
    
    preset = args.preset
    print(f"\nPreset: {preset}")
    print(f"Device: {device}")
    
    model = FastMatchStereo(args, preset=preset)
    
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
    
    model.to(device).eval()
    
    results = []
    
    for h, w in resolutions:
        print(f"\n  {h}x{w}...")
        
        try:
            stats = benchmark_pytorch_model(
                model, h, w, device, dtype,
                num_warmup=2, num_runs=5
            )
            
            result = {
                'resolution': f"{h}x{w}",
                'height': h,
                'width': w,
                'pixels': h * w,
                **stats,
            }
            results.append(result)
            
            print(f"    Time: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
            
        except Exception as e:
            print(f"    Failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("RESOLUTION SCALING SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Resolution':<12} {'Pixels':<12} {'Time (ms)':<15} {'ms/megapixel':<15}")
    print("-" * 55)
    
    for r in results:
        mpx_time = r['mean_ms'] / (r['pixels'] / 1e6)
        print(f"{r['resolution']:<12} {r['pixels']:<12} {r['mean_ms']:.2f}           {mpx_time:.2f}")
    
    return results


def run_full_benchmark(args):
    """Run all benchmark modes and generate report."""
    print("\n" + "=" * 80)
    print("FULL BENCHMARK SUITE")
    print("=" * 80)
    
    all_results = {
        'system_info': get_system_info(),
        'config': {
            'height': args.height,
            'width': args.width,
            'variant': args.variant,
            'precision': args.precision,
        }
    }
    
    # Benchmark presets
    all_results['presets'] = benchmark_presets(args)
    
    # Benchmark resolutions
    all_results['resolutions'] = benchmark_resolutions(args)
    
    # PyTorch vs ONNX (if ONNX models available)
    if args.onnx_dir and os.path.exists(args.onnx_dir):
        all_results['pytorch_vs_onnx'] = benchmark_pytorch_vs_onnx(args)
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n\nFull results saved to: {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MatchStereo models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model configuration
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to PyTorch checkpoint')
    parser.add_argument('--variant', choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--preset', type=str, default='fast',
                        choices=list(OPTIMIZATION_PRESETS.keys()))
    parser.add_argument('--mat_impl', choices=['pytorch', 'cuda'], default='pytorch')
    
    # Input settings
    parser.add_argument('--height', type=int, default=384)
    parser.add_argument('--width', type=int, default=384)
    
    # Runtime
    parser.add_argument('--device_id', type=int, default=-1,
                        help='GPU device ID, -1 for CPU')
    parser.add_argument('--precision', choices=['fp32', 'fp16'], default='fp32')
    
    # Benchmark settings
    parser.add_argument('--mode', choices=['presets', 'comparison', 'resolutions', 'full'],
                        default='presets', help='Benchmark mode')
    parser.add_argument('--num_warmup', type=int, default=3)
    parser.add_argument('--num_runs', type=int, default=10)
    
    # ONNX
    parser.add_argument('--onnx_dir', type=str, default='exports',
                        help='Directory containing ONNX models')
    
    # Output
    parser.add_argument('--output', type=str,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Validate input size
    if args.height % 32 != 0 or args.width % 32 != 0:
        args.height = (args.height // 32) * 32
        args.width = (args.width // 32) * 32
        print(f"Adjusted resolution to: {args.height}x{args.width}")
    
    # Print system info
    info = get_system_info()
    print("\nSystem Information:")
    print(f"  CPU: {info['cpu']['count']} cores, {info['cpu']['threads']} threads")
    print(f"  Memory: {info['memory_gb']:.1f} GB")
    print(f"  PyTorch: {info['pytorch_version']}")
    if info['cuda']['available']:
        print(f"  CUDA: {info['cuda']['version']} ({info['cuda']['device']})")
    else:
        print("  CUDA: Not available")
    if info.get('onnxruntime_version'):
        print(f"  ONNX Runtime: {info['onnxruntime_version']}")
    
    # Run benchmark
    if args.mode == 'presets':
        benchmark_presets(args)
    elif args.mode == 'comparison':
        benchmark_pytorch_vs_onnx(args)
    elif args.mode == 'resolutions':
        benchmark_resolutions(args)
    elif args.mode == 'full':
        run_full_benchmark(args)


if __name__ == "__main__":
    main()

