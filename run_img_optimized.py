#!/usr/bin/env python3
"""
Optimized MatchStereo/MatchFlow inference script with configurable speed presets.

This script extends run_img.py with support for optimization presets,
making it suitable for embedded deployment on devices like Raspberry Pi 4.

Usage:
    # List available presets
    python run_img_optimized.py --list_presets
    
    # Run with specific preset
    python run_img_optimized.py --preset fast --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --img0_dir images/left/ --img1_dir images/right/ --output_path outputs
    
    # Benchmark all presets
    python run_img_optimized.py --benchmark --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \
        --img0_dir images/left/ --img1_dir images/right/
"""

import argparse
import os
import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

from dataloader.stereo import transforms
from utils.utils import InputPadder, calc_noc_mask
from utils.file_io import write_pfm
from models.fast_match_stereo import FastMatchStereo, OPTIMIZATION_PRESETS, list_presets

torch.backends.cudnn.benchmark = True


def visualize_disparity(disparity):
    """
    Convert disparity map to grayscale visualization (same as gradio_app.py).
    Uses min-max normalization: brighter = higher disparity (closer).
    
    Args:
        disparity: 2D numpy array of disparity values
        
    Returns:
        numpy array (H, W) uint8 grayscale disparity image
    """
    disp = disparity.copy()
    
    min_val = disp.min()
    max_val = disp.max()
    
    if max_val - min_val > 1e-6:
        disp_norm = (disp - min_val) / (max_val - min_val)
    else:
        disp_norm = np.zeros_like(disp)
    
    disp_img = (disp_norm * 255).astype(np.uint8)
    return disp_img


def save_disparity_image(disparity, save_path):
    """
    Save disparity map as grayscale JPEG using OpenCV.
    
    Args:
        disparity: 2D numpy array of disparity values
        save_path: Path to save the image (should end in .jpg)
    """
    disp_img = visualize_disparity(disparity)
    cv2.imwrite(save_path, disp_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return disp_img


def save_disparity_comparison(left_img, disparity, save_path, preset_name, inference_time_ms=None):
    """
    Save a side-by-side comparison of input image and disparity map using OpenCV.
    
    Args:
        left_img: Original left image (numpy array RGB or BGR)
        disparity: 2D numpy array of disparity values
        save_path: Path to save the comparison image (JPEG)
        preset_name: Name of the preset used
        inference_time_ms: Optional inference time in milliseconds
    """
    # Convert left image to BGR if needed (assuming input is RGB)
    if len(left_img.shape) == 3 and left_img.shape[2] == 3:
        left_bgr = cv2.cvtColor(left_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        left_bgr = left_img.astype(np.uint8)
    
    # Get grayscale disparity and convert to BGR for display
    disp_gray = visualize_disparity(disparity)
    disp_bgr = cv2.cvtColor(disp_gray, cv2.COLOR_GRAY2BGR)
    
    # Create side-by-side comparison
    h, w = left_bgr.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = left_bgr
    comparison[:, w:] = disp_bgr
    
    # Add labels
    label_text = f"Preset: {preset_name}"
    if inference_time_ms is not None:
        label_text += f" | {inference_time_ms:.0f}ms"
    
    cv2.putText(comparison, label_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, label_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
    
    cv2.putText(comparison, "Input", (w//2 - 30, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(comparison, "Disparity", (w + w//2 - 50, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite(save_path, comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return comparison


def run_frame(model, left, right, stereo, low_res_init, factor=2.):
    """Run inference on a single frame pair."""
    if low_res_init:
        left_ds = F.interpolate(left, scale_factor=1/factor, mode='bilinear', align_corners=True)
        right_ds = F.interpolate(right, scale_factor=1/factor, mode='bilinear', align_corners=True)
        padder_ds = InputPadder(left_ds.shape, padding_factor=32)
        left_ds, right_ds = padder_ds.pad(left_ds, right_ds)

        field_up_ds = model(left_ds, right_ds, stereo=stereo)['field_up']
        field_up_ds = padder_ds.unpad(field_up_ds.permute(0, 3, 1, 2).contiguous()).contiguous()
        field_up_init = F.interpolate(field_up_ds, scale_factor=factor/32, mode='bilinear', align_corners=True) * (factor/32)
        field_up_init = field_up_init.permute(0, 2, 3, 1).contiguous()
        results_dict = model(left, right, stereo=stereo, init_flow=field_up_init)
    else:
        results_dict = model(left, right, stereo=stereo)

    return results_dict


def benchmark_presets(args, left_names, right_names, val_transform, device, dtype):
    """Benchmark all available presets and print comparison."""
    print("\n" + "=" * 80)
    print("BENCHMARKING ALL OPTIMIZATION PRESETS")
    print("=" * 80)
    
    # Use first image pair for benchmarking
    left = np.array(Image.open(left_names[0]).convert('RGB')).astype(np.float32)
    right = np.array(Image.open(right_names[0]).convert('RGB')).astype(np.float32)
    
    sample = {'left': left, 'right': right}
    sample = val_transform(sample)
    left_tensor = sample['left'].to(device, dtype=dtype).unsqueeze(0)
    right_tensor = sample['right'].to(device, dtype=dtype).unsqueeze(0)
    
    if args.inference_size is not None:
        left_tensor = F.interpolate(left_tensor, size=args.inference_size, mode='bilinear', align_corners=True)
        right_tensor = F.interpolate(right_tensor, size=args.inference_size, mode='bilinear', align_corners=True)
    else:
        padder = InputPadder(left_tensor.shape, padding_factor=32)
        left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    
    print(f"\nInput resolution: {left_tensor.shape}")
    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print("\n" + "-" * 80)
    
    results = []
    stereo = (args.mode == 'stereo')
    
    for preset_name in OPTIMIZATION_PRESETS.keys():
        print(f"\nTesting preset: {preset_name}")
        
        # Create model with this preset
        model = FastMatchStereo(args, preset=preset_name)
        
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
        
        model.to(device).to(dtype).eval()
        
        config_summary = model.get_config_summary()
        
        # Warmup
        with torch.inference_mode():
            for _ in range(2):
                _ = model(left_tensor, right_tensor, stereo=stereo)
        
        # Benchmark
        times = []
        num_runs = 5
        
        with torch.inference_mode():
            for _ in range(num_runs):
                if device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                _ = run_frame(model, left_tensor, right_tensor, stereo, args.low_res_init)
                
                if device != 'cpu' and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.append({
            'preset': preset_name,
            'avg_time_ms': avg_time,
            'std_time_ms': std_time,
            'iterations': config_summary['total_iterations'],
            'description': config_summary['description'],
        })
        
        print(f"  Iterations: {config_summary['total_iterations']}")
        print(f"  Avg time: {avg_time:.2f} ± {std_time:.2f} ms")
        
        # Clear memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\n{'Preset':<15} {'Iterations':<12} {'Time (ms)':<15} {'Speedup':<10} Description")
    print("-" * 80)
    
    baseline_time = results[0]['avg_time_ms']  # 'default' is first
    
    for r in results:
        speedup = baseline_time / r['avg_time_ms']
        print(f"{r['preset']:<15} {r['iterations']:<12} {r['avg_time_ms']:.2f} ± {r['std_time_ms']:.2f}    {speedup:.2f}x       {r['description'][:30]}")
    
    print("=" * 80)
    
    return results


def run(args):
    """Run MatchStereo/MatchFlow on stereo/flow pairs with optimization presets."""
    
    # List presets if requested
    if args.list_presets:
        list_presets()
        return
    
    stereo = (args.mode == 'stereo')
    val_transform_list = [
        transforms.Resize(scale_x=args.scale, scale_y=args.scale),
        transforms.ToTensor(no_normalize=True)
    ]
    val_transform = transforms.Compose(val_transform_list)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() and args.device_id >= 0 else 'cpu'
    dtypes = {'fp32': torch.float, 'fp16': torch.half, 'bf16': torch.bfloat16}
    dtype = dtypes[args.precision]

    # Get image paths
    if args.middv3_dir is not None:
        left_names = sorted(glob(args.middv3_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.middv3_dir + '/*/*/im1.png'))
    elif args.eth3d_dir is not None:
        left_names = sorted(glob(args.eth3d_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.eth3d_dir + '/*/*/im1.png'))
    else:
        left_names = sorted(glob(args.img0_dir + '/*.png') + glob(args.img0_dir + '/*.jpg') + glob(args.img0_dir + '/*.bmp'))
        right_names = sorted(glob(args.img1_dir + '/*.png') + glob(args.img1_dir + '/*.jpg') + glob(args.img1_dir + '/*.bmp'))
    
    assert len(left_names) == len(right_names), f"Mismatched image pairs: {len(left_names)} vs {len(right_names)}"
    
    num_samples = len(left_names)
    print(f'{num_samples} test samples found')
    
    # Benchmark mode
    if args.benchmark:
        benchmark_presets(args, left_names, right_names, val_transform, device, dtype)
        return

    # Create model with specified preset
    print(f"\nCreating model with preset: {args.preset}")
    model = FastMatchStereo(args, preset=args.preset)
    
    # Print configuration
    config = model.get_config_summary()
    print(f"  Description: {config['description']}")
    print(f"  Total iterations: {config['total_iterations']}")
    print(f"  Estimated speedup: {config['estimated_speedup']}x")
    
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        if missing:
            print(f"  Warning: {len(missing)} missing keys (expected for reduced configs)")
        if unexpected:
            print(f"  Warning: {len(unexpected)} unexpected keys")
    
    model.to(device)
    model.eval()
    model = model.to(dtype)
    
    if torch.cuda.is_available() and not args.no_compile and args.device_id >= 0:
        print('Compiling the model, this may take several minutes...')
        torch.backends.cuda.matmul.allow_tf32 = True
        model = torch.compile(model, dynamic=False)

    # Setup timing
    if torch.cuda.is_available() and args.device_id >= 0:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    else:
        args.test_inference_time = args.test_inference_time  # Will use CPU timing

    total_time = 0
    
    for i in range(num_samples):
        left = np.array(Image.open(left_names[i]).convert('RGB')).astype(np.float32)
        right = np.array(Image.open(right_names[i]).convert('RGB')).astype(np.float32)

        sample = {'left': left, 'right': right}
        sample = val_transform(sample)
        left = sample['left'].to(device, dtype=dtype).unsqueeze(0)
        right = sample['right'].to(device, dtype=dtype).unsqueeze(0)

        if args.inference_size is None:
            padder = InputPadder(left.shape, padding_factor=32)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]
            left = F.interpolate(left, size=args.inference_size, mode='bilinear', align_corners=True)
            right = F.interpolate(right, size=args.inference_size, mode='bilinear', align_corners=True)

        print(f"Processing [{i+1}/{num_samples}]: {os.path.basename(left_names[i])} - Resolution: {left.shape[-2:]}")
        
        with torch.inference_mode():
            if args.test_inference_time:
                # Warmup
                for _ in range(3):
                    _ = model(left, right, stereo=stereo)

                if torch.cuda.is_available() and args.device_id >= 0:
                    start_event.record()
                    for _ in range(5):
                        results_dict = run_frame(model, left, right, stereo, args.low_res_init)
                    end_event.record()
                    end_event.synchronize()
                    inference_time = start_event.elapsed_time(end_event) / 5
                else:
                    # CPU timing
                    start = time.perf_counter()
                    for _ in range(5):
                        results_dict = run_frame(model, left, right, stereo, args.low_res_init)
                    end = time.perf_counter()
                    inference_time = (end - start) * 1000 / 5

                print(f"  Inference Time: {inference_time:.2f} ms")
                total_time += inference_time
            else:
                start = time.perf_counter()
                results_dict = run_frame(model, left, right, stereo, args.low_res_init)
                end = time.perf_counter()
                inference_time = (end - start) * 1000
                print(f"  Inference Time: {inference_time:.2f} ms")
                total_time += inference_time

            field_up = results_dict['field_up'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = results_dict['self_rpos'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = F.interpolate(self_rpos, scale_factor=4, mode='bilinear', align_corners=True) * 4
            
            if args.inference_size is None:
                field_up = padder.unpad(field_up)
                self_rpos = padder.unpad(self_rpos)
            else:
                field_up = F.interpolate(field_up, size=ori_size, mode='bilinear', align_corners=True)
                field_up[:, 0] = field_up[:, 0] * (ori_size[1] / float(args.inference_size[1]))
                field_up[:, 1] = field_up[:, 1] * (ori_size[0] / float(args.inference_size[0]))

                self_rpos = F.interpolate(self_rpos, size=ori_size, mode='bilinear', align_corners=True)
                self_rpos[:, 0] = self_rpos[:, 0] * ori_size[1] / float(args.inference_size[1])
                self_rpos[:, 1] = self_rpos[:, 1] * ori_size[0] / float(args.inference_size[0])

        # Save outputs
        if args.middv3_dir is not None:
            save_name = left_names[i].replace('/MiddEval3', '/MiddEval3_results').replace('/im0.png', '/disp0MatchStereo.pfm')
        elif args.eth3d_dir is not None:
            parts = list(Path(left_names[i]).parts)
            parts[1] = "ETH3D_results"
            parts[2] = "low_res_two_view"
            save_name = str(Path(*parts[:3]) / f"{parts[3]}.pfm")
        else:
            save_name = os.path.join(args.output_path, os.path.basename(left_names[i])[:-4] + f'_{args.mode}_{args.preset}.pfm')
        
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        noc_mask = calc_noc_mask(field_up.permute(0, 2, 3, 1), A=8)
        noc_mask = noc_mask[0].detach().cpu().numpy()
        noc_mask = np.where(noc_mask, 255, 128).astype(np.uint8)
        noc_img = Image.fromarray(noc_mask)
        noc_img.save(save_name[:-4] + '_noc.png')
        
        field_up = torch.cat((field_up, torch.zeros_like(field_up[:, :1])), dim=1)
        field_up = field_up.permute(0, 2, 3, 1).contiguous()
        field, field_r = field_up.chunk(2, dim=0)
        
        if stereo:
            field = (-field[..., 0]).clamp(min=0)
            field_r = field_r[..., 0].clamp(min=0)
        
        field = field[0].detach().cpu().numpy()
        field_r = field_r[0].detach().cpu().numpy()
        write_pfm(save_name, field)
        
        # Save grayscale disparity visualization as JPEG (same as gradio_app)
        disp_path = save_name[:-4] + '.jpg'
        save_disparity_image(field, disp_path)
        print(f"  Saved: {disp_path}")
        
        # Save side-by-side comparison if requested
        if args.save_comparison:
            left_original = cv2.imread(left_names[i])
            left_original = cv2.cvtColor(left_original, cv2.COLOR_BGR2RGB)
            comparison_path = save_name[:-4] + '_comparison.jpg'
            save_disparity_comparison(left_original, field, comparison_path, args.preset, inference_time)
            print(f"  Saved: {comparison_path}")
        
        if args.save_right:
            write_pfm(save_name[:-4] + '_r.pfm', field_r)
            save_disparity_image(field_r, save_name[:-4] + '_r.jpg')

        if args.save_rpos:
            self_rpos, _ = self_rpos.chunk(2, dim=0)
            self_rpos = self_rpos[0].detach().cpu().numpy()
            write_pfm(save_name[:-4] + '_self_rpos_x.pfm', self_rpos[0])
            write_pfm(save_name[:-4] + '_self_rpos_y.pfm', self_rpos[1])

    # Print summary
    print("\n" + "=" * 60)
    print(f"INFERENCE COMPLETE")
    print(f"  Preset: {args.preset}")
    print(f"  Samples processed: {num_samples}")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Average time per sample: {total_time / num_samples:.2f} ms")
    print("=" * 60)


def main():
    """Run MatchStereo/MatchFlow inference with optimization presets."""
    parser = argparse.ArgumentParser(
        description="Optimized MatchStereo/MatchFlow inference with configurable speed presets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available presets
  python run_img_optimized.py --list_presets
  
  # Run with fast preset
  python run_img_optimized.py --preset fast \\
      --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \\
      --img0_dir images/left/ --img1_dir images/right/
  
  # Benchmark all presets
  python run_img_optimized.py --benchmark \\
      --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \\
      --img0_dir images/left/ --img1_dir images/right/
  
  # Run optimized for Raspberry Pi 4
  python run_img_optimized.py --preset rpi4_optimized --device_id -1 \\
      --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth \\
      --img0_dir images/left/ --img1_dir images/right/
"""
    )
    
    # Model configuration
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint')
    parser.add_argument('--preset', type=str, default='default',
                        choices=list(OPTIMIZATION_PRESETS.keys()),
                        help='Optimization preset (default: default)')
    parser.add_argument('--variant', choices=['tiny', 'small', 'base'], default='tiny',
                        help='Model variant (default: tiny)')
    parser.add_argument('--mode', choices=['stereo', 'flow'], default='stereo',
                        help='Task mode (default: stereo)')
    
    # Input/output
    parser.add_argument('--img0_dir', default=None, type=str, help='Left/reference image directory')
    parser.add_argument('--img1_dir', default=None, type=str, help='Right/target image directory')
    parser.add_argument('--middv3_dir', default=None, type=str, help='Middlebury v3 dataset directory')
    parser.add_argument('--eth3d_dir', default=None, type=str, help='ETH3D dataset directory')
    parser.add_argument('--output_path', default='outputs', type=str, help='Output directory')
    
    # Runtime settings
    parser.add_argument('--device_id', default=0, type=int, help='GPU device ID, -1 for CPU')
    parser.add_argument('--scale', default=1, type=float, help='Input scaling factor')
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='Inference resolution [H, W], must be divisible by 32')
    parser.add_argument('--mat_impl', choices=['pytorch', 'cuda'], default='pytorch',
                        help='MatchAttention implementation (default: pytorch)')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32',
                        help='Precision mode (default: fp32)')
    parser.add_argument('--no_compile', action='store_true', default=False,
                        help='Disable torch.compile')
    parser.add_argument('--low_res_init', action='store_true', default=False,
                        help='Use low-resolution init for high-res images (>=2K)')
    
    # Output options
    parser.add_argument('--save_right', action='store_true', default=False,
                        help='Save right/target view disparity/flow')
    parser.add_argument('--save_rpos', action='store_true', default=False,
                        help='Save self relative positions')
    parser.add_argument('--save_comparison', action='store_true', default=False,
                        help='Save side-by-side comparison of input and disparity')
    
    # Testing/benchmarking
    parser.add_argument('--test_inference_time', action='store_true', default=False,
                        help='Run multiple iterations and report average time')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Benchmark all presets and print comparison')
    parser.add_argument('--list_presets', action='store_true', default=False,
                        help='List available optimization presets and exit')

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

