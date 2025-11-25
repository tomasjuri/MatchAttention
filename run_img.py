import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image
from glob import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from dataloader.stereo import transforms
from utils.utils import InputPadder, calc_noc_mask
from utils.file_io import write_pfm
from models.match_stereo import MatchStereo

torch.backends.cudnn.benchmark = True


def is_onnx_model(checkpoint_path):
    """Check if the checkpoint is an ONNX model based on file extension."""
    return checkpoint_path.lower().endswith('.onnx')


def run_onnx(args):
    """Run inference using ONNX Runtime"""
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required for ONNX inference. Install with: pip install onnxruntime")
    
    stereo = (args.mode == 'stereo')
    if not stereo:
        raise NotImplementedError("ONNX inference currently only supports stereo mode")
    
    val_transform_list = [transforms.Resize(scale_x=args.scale, scale_y=args.scale), 
                          transforms.ToTensor(no_normalize=True)]
    val_transform = transforms.Compose(val_transform_list)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Setup ONNX Runtime session
    providers = ['CPUExecutionProvider']
    if args.device_id >= 0:
        try:
            # Try CUDA provider if GPU requested
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except:
            print("CUDA provider not available, falling back to CPU")
            providers = ['CPUExecutionProvider']
    
    print(f"Loading ONNX model from: {args.checkpoint_path}")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(args.checkpoint_path, session_options, providers=providers)
    
    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"ONNX Inputs: {input_names}")
    print(f"ONNX Outputs: {output_names}")

    # Gather image paths
    if args.middv3_dir is not None:
        left_names = sorted(glob(args.middv3_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.middv3_dir + '/*/*/im1.png'))
    elif args.eth3d_dir is not None:
        left_names = sorted(glob(args.eth3d_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.eth3d_dir + '/*/*/im1.png'))
    else:
        left_names = sorted(glob(args.img0_dir + '/*.png') + glob(args.img0_dir + '/*.jpg') + glob(args.img0_dir + '/*.bmp'))
        right_names = sorted(glob(args.img1_dir + '/*.png') + glob(args.img1_dir + '/*.jpg') + glob(args.img1_dir + '/*.bmp'))
    assert len(left_names) == len(right_names)

    num_samples = len(left_names)
    print(f'{num_samples} test samples found')

    for i in range(num_samples):
        left = np.array(Image.open(left_names[i]).convert('RGB')).astype(np.float32)
        right = np.array(Image.open(right_names[i]).convert('RGB')).astype(np.float32)

        sample = {'left': left, 'right': right}
        sample = val_transform(sample)
        
        # Convert to numpy for ONNX (keep in [0, 255] range as expected by ONNX model)
        left_tensor = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        right_tensor = sample['right'].unsqueeze(0)  # [1, 3, H, W]
        
        ori_size = left_tensor.shape[-2:]
        
        # Pad to be divisible by 32
        if args.inference_size is None:
            padder = InputPadder(left_tensor.shape, padding_factor=32)
            left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
        else:
            left_tensor = F.interpolate(left_tensor, size=args.inference_size, mode='bilinear', align_corners=True)
            right_tensor = F.interpolate(right_tensor, size=args.inference_size, mode='bilinear', align_corners=True)
        
        # Convert to numpy arrays for ONNX Runtime
        left_np = left_tensor.numpy().astype(np.float32)
        right_np = right_tensor.numpy().astype(np.float32)

        print(f"Resolution: {left_np.shape}")
        
        # Run ONNX inference
        if args.test_inference_time:
            # Warmup
            for _ in range(5):
                _ = session.run(output_names, {input_names[0]: left_np, input_names[1]: right_np})
            
            # Benchmark
            times = []
            for _ in range(5):
                start = time.perf_counter()
                outputs = session.run(output_names, {input_names[0]: left_np, input_names[1]: right_np})
                end = time.perf_counter()
                times.append((end - start) * 1000)
            
            inference_time = np.mean(times)
            print(f"Inference Time (ONNX) on {left_names[i]}: {inference_time:.6f} ms")
            disparity = outputs[0]
        else:
            outputs = session.run(output_names, {input_names[0]: left_np, input_names[1]: right_np})
            disparity = outputs[0]  # [B, H, W]
        
        # Convert to torch tensor for post-processing
        field = torch.from_numpy(disparity)  # [B, H, W]
        
        # Unpad or rescale
        if args.inference_size is None:
            # Unpad - add channel dim for unpadding then remove
            field = field.unsqueeze(1)  # [B, 1, H, W]
            field = padder.unpad(field)
            field = field.squeeze(1)  # [B, H, W]
        else:
            field = field.unsqueeze(1)  # [B, 1, H, W]
            field = F.interpolate(field, size=ori_size, mode='bilinear', align_corners=True)
            field = field * (ori_size[1] / float(args.inference_size[1]))
            field = field.squeeze(1)  # [B, H, W]
        
        field = field[0].numpy()  # [H, W]
        
        # Save outputs
        if args.middv3_dir is not None:
            save_name = left_names[i].replace('/MiddEval3', '/MiddEval3_results').replace('/im0.png', '/disp0MatchStereo.pfm')
        elif args.eth3d_dir is not None:
            parts = list(Path(left_names[i]).parts)
            parts[1] = "ETH3D_results"
            parts[2] = "low_res_two_view"
            save_name = str(Path(*parts[:3]) / f"{parts[3]}.pfm")
        else:
            save_name = os.path.join(args.output_path, os.path.basename(left_names[i])[:-4] + f'_{args.mode}.pfm')
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        write_pfm(save_name, field)

        # Save PNG visualization with colormap
        min_val = field.min()
        max_val = field.max()
        if max_val - min_val > 1e-6:
            disparity_norm = (field - min_val) / (max_val - min_val)
        else:
            disparity_norm = np.zeros_like(field)
        
        colormap = plt.get_cmap('turbo')
        disparity_colored = colormap(disparity_norm)
        disparity_rgb = (disparity_colored[:, :, :3] * 255).astype(np.uint8)
        disp_vis = Image.fromarray(disparity_rgb)
        disp_vis.save(save_name[:-4] + '.jpg', quality=95)

        if args.test_inference_time:
            if args.middv3_dir is not None:
                save_time_name = save_name.replace('/disp0MatchStereo.pfm', '/timeMatchStereo.txt')
                with open(save_time_name, 'w') as f:
                    f.write(str(inference_time / 1000))
            elif args.eth3d_dir is not None:
                save_time_name = save_name.replace('.pfm', '.txt')
                with open(save_time_name, 'w') as f:
                    f.write(str(f"runtime {inference_time / 1000}"))

    print("ONNX Inference done.")

def run_frame(model, left, right, stereo, low_res_init, factor=2.):
    if low_res_init: # downsample to 1/2, can also be 1/4
        left_ds = F.interpolate(left, scale_factor=1/factor, mode='bilinear', align_corners=True)
        right_ds = F.interpolate(right, scale_factor=1/factor, mode='bilinear', align_corners=True)
        padder_ds = InputPadder(left_ds.shape, padding_factor=32)
        left_ds, right_ds = padder_ds.pad(left_ds, right_ds)

        field_up_ds = model(left_ds, right_ds, stereo=stereo)['field_up']
        field_up_ds = padder_ds.unpad(field_up_ds.permute(0, 3, 1, 2).contiguous()).contiguous()
        field_up_init = F.interpolate(field_up_ds, scale_factor=factor/32, mode='bilinear', align_corners=True)*(factor/32) # init resolution 1/32
        field_up_init = field_up_init.permute(0, 2, 3, 1).contiguous()
        results_dict = model(left, right, stereo=stereo, init_flow=field_up_init)
    else:
        results_dict = model(left, right, stereo=stereo)

    return results_dict

def run(args):
    """Run MatchStereo/MatchFlow on stereo/flow pairs"""
    stereo = (args.mode == 'stereo')
    val_transform_list = [transforms.Resize(scale_x=args.scale, scale_y=args.scale), 
                          transforms.ToTensor(no_normalize=True)]
    val_transform = transforms.Compose(val_transform_list)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    device = torch.device(f'cuda:{args.device_id}') if torch.cuda.is_available() and args.device_id >=0 else 'cpu'
    dtypes = {'fp32': torch.float, 'fp16': torch.half, 'bf16': torch.bfloat16}
    dtype = dtypes[args.precision]

    model = MatchStereo(args)
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict=checkpoint['model'], strict=False)
    model.to(device)
    model.eval()
    model = model.to(dtype)
    if torch.cuda.is_available() and not args.no_compile and args.device_id >=0:
        print('compiling the model, this may take several minutes')
        torch.backends.cuda.matmul.allow_tf32 = True
        model = torch.compile(model, dynamic=False)

    if args.middv3_dir is not None:
        left_names = sorted(glob(args.middv3_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.middv3_dir + '/*/*/im1.png'))
    elif args.eth3d_dir is not None:
        left_names = sorted(glob(args.eth3d_dir + '/*/*/im0.png'))
        right_names = sorted(glob(args.eth3d_dir + '/*/*/im1.png'))
    else:
        left_names = sorted(glob(args.img0_dir + '/*.png') + glob(args.img0_dir + '/*.jpg') + glob(args.img0_dir + '/*.bmp'))
        right_names = sorted(glob(args.img1_dir + '/*.png') + glob(args.img1_dir + '/*.jpg') + glob(args.img1_dir + '/*.bmp'))
    assert len(left_names) == len(right_names)

    num_samples = len(left_names)
    print('%d test samples found' % num_samples)

    if torch.cuda.is_available() and args.device_id >=0:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
    else:
        args.test_inference_time = False

    for i in range(num_samples):

        left = np.array(Image.open(left_names[i]).convert('RGB')).astype(np.float32)
        right = np.array(Image.open(right_names[i]).convert('RGB')).astype(np.float32)

        sample = {'left': left, 'right': right}
        sample = val_transform(sample)
        left = sample['left'].to(device, dtype=dtype).unsqueeze(0) # [1, 3, H, W]
        right = sample['right'].to(device, dtype=dtype).unsqueeze(0) # [1, 3, H, W]

        if args.inference_size is None:
            padder = InputPadder(left.shape, padding_factor=32)
            left, right = padder.pad(left, right)
        else:
            ori_size = left.shape[-2:]

            left = F.interpolate(left, size=args.inference_size, mode='bilinear', align_corners=True)
            right = F.interpolate(right, size=args.inference_size, mode='bilinear', align_corners=True)

        print("Resolution: ", left.shape)
        with torch.inference_mode():
            if args.test_inference_time:
                for _ in range(5): # warmup
                    _ = model(left, right, stereo=stereo)

                start_event.record()
                for _ in range(5):
                    results_dict = run_frame(model, left, right, stereo, args.low_res_init)
                end_event.record()
                end_event.synchronize()

                inference_time = start_event.elapsed_time(end_event) / 5  # in milliseconds
                print(f"Inference Time (GPU) on {left_names[i]}: {inference_time:.6f} ms")

            else:
                results_dict = run_frame(model, left, right, stereo, args.low_res_init)

            field_up = results_dict['field_up'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = results_dict['self_rpos'].permute(0, 3, 1, 2).float().contiguous()
            self_rpos = F.interpolate(self_rpos, scale_factor=4, mode='bilinear', align_corners=True)*4
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

        if args.middv3_dir is not None:
            save_name = left_names[i].replace('/MiddEval3', '/MiddEval3_results').replace('/im0.png', '/disp0MatchStereo.pfm')
        elif args.eth3d_dir is not None:
            parts = list(Path(left_names[i]).parts)
            parts[1] = "ETH3D_results"
            parts[2] = "low_res_two_view"
            save_name = str(Path(*parts[:3]) / f"{parts[3]}.pfm")
        else:
            save_name = os.path.join(args.output_path, os.path.basename(left_names[i])[:-4] + f'_{args.mode}.pfm')
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

        noc_mask = calc_noc_mask(field_up.permute(0, 2, 3, 1), A=8)
        ## field[~noc_mask] = torch.inf # NOTE: can filter out un-reliable matches by consistency check
        noc_mask = noc_mask[0].detach().cpu().numpy()
        noc_mask = np.where(noc_mask, 255, 128).astype(np.uint8)
        noc_img = Image.fromarray(noc_mask)
        noc_img.save(save_name[:-4] + '_noc.jpg', quality=95)
        field_up = torch.cat((field_up, torch.zeros_like(field_up[:, :1])), dim=1)
        field_up = field_up.permute(0, 2, 3, 1).contiguous() # [B, H, W, 3]
        field, field_r = field_up.chunk(2, dim=0)
        if stereo:
            field = (-field[..., 0]).clamp(min=0)
            field_r = field_r[..., 0].clamp(min=0)
        field = field[0].detach().cpu().numpy()
        field_r = field_r[0].detach().cpu().numpy()
        write_pfm(save_name, field)
        if args.save_right:
            write_pfm(save_name[:-4] + '_r.pfm', field_r)

        # Save PNG visualization (same as Gradio app with colormap)
        if stereo:
            # Normalize disparity to 0-1 range
            min_val = field.min()
            max_val = field.max()
            if max_val - min_val > 1e-6:
                disparity_norm = (field - min_val) / (max_val - min_val)
            else:
                disparity_norm = np.zeros_like(field)
            
            # Apply turbo colormap (commonly used for disparity/depth visualization)
            # This matches how Gradio displays 2D arrays with automatic colormap
            colormap = plt.get_cmap('turbo')
            disparity_colored = colormap(disparity_norm)
            # Convert from float [0,1] to uint8 [0,255] and remove alpha channel
            disparity_rgb = (disparity_colored[:, :, :3] * 255).astype(np.uint8)
            disp_vis = Image.fromarray(disparity_rgb)
            disp_vis.save(save_name[:-4] + '.jpg', quality=95)
            
            if args.save_right:
                min_val_r = field_r.min()
                max_val_r = field_r.max()
                if max_val_r - min_val_r > 1e-6:
                    disparity_norm_r = (field_r - min_val_r) / (max_val_r - min_val_r)
                else:
                    disparity_norm_r = np.zeros_like(field_r)
                
                # Apply turbo colormap to right disparity
                disparity_colored_r = colormap(disparity_norm_r)
                disparity_rgb_r = (disparity_colored_r[:, :, :3] * 255).astype(np.uint8)
                disp_vis_r = Image.fromarray(disparity_rgb_r)
                disp_vis_r.save(save_name[:-4] + '_r.jpg', quality=95)
        else:
            # Optical flow visualization (same as gradio_app.py)
            import cv2
            u = field[..., 0] if len(field.shape) == 3 else field[0]
            v = field[..., 1] if len(field.shape) == 3 else field[1]
            
            rad = np.sqrt(u**2 + v**2)
            rad_max = np.max(rad)
            epsilon = 1e-8
            
            if rad_max > epsilon:
                u = u / (rad_max + epsilon)
                v = v / (rad_max + epsilon)
            
            h, w = u.shape
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            
            mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            flow_vis = Image.fromarray(flow_rgb)
            flow_vis.save(save_name[:-4] + '.jpg', quality=95)

        if args.save_rpos:
            self_rpos, _ = self_rpos.chunk(2, dim=0)
            self_rpos = self_rpos[0].detach().cpu().numpy()
            write_pfm(save_name[:-4] + '_self_rpos_x.pfm', self_rpos[0])
            write_pfm(save_name[:-4] + '_self_rpos_y.pfm', self_rpos[1])

        if args.test_inference_time:
            if args.middv3_dir is not None:
                save_time_name = save_name.replace('/disp0MatchStereo.pfm', '/timeMatchStereo.txt')
                with open(save_time_name, 'w') as f:
                    f.write(str(inference_time / 1000))
            elif args.eth3d_dir is not None:
                save_time_name = save_name.replace('.pfm', '.txt')
                with open(save_time_name, 'w') as f:
                    f.write(str(f"runtime {inference_time / 1000}"))

    print("Inference done.")

def main():
    """Run MatchStereo/MatchFlow inference example"""
    parser = argparse.ArgumentParser(
        description="Inference scripts of MatchStereo/MatchFlow with PyTorch or ONNX models."
    )
    parser.add_argument('--checkpoint_path', required=True, type=str, 
                        help='Path to MatchStereo/MatchFlow checkpoint (.pth for PyTorch, .onnx for ONNX)')
    parser.add_argument('--mode', choices=['stereo', 'flow'], default='stereo', help='Support stereo and flow tasks')
    parser.add_argument('--img0_dir', default=None, type=str, help='Reference view')
    parser.add_argument('--img1_dir', default=None, type=str, help='Target view')
    parser.add_argument('--middv3_dir', default=None, type=str)
    parser.add_argument('--eth3d_dir', default=None, type=str)
    parser.add_argument('--output_path', default='outputs', type=str)
    parser.add_argument('--device_id', default=0, type=int, help='Device id of gpu, -1 for cpu')
    parser.add_argument('--scale', default=1, type=float)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+', help='Shall be divisible by 32')
    parser.add_argument('--mat_impl', choices=['pytorch', 'cuda'], default='pytorch', help='MatchAttention implementation')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--variant', choices=['tiny', 'small', 'base'], default='tiny')
    parser.add_argument('--no_compile', action='store_true', default=False, help='Disable torch.compile')
    parser.add_argument('--test_inference_time', action='store_true', default=False)
    parser.add_argument('--save_right', action='store_true', default=False, help='Save the right/target view disp/flow')
    parser.add_argument('--save_rpos', action='store_true', default=False, help='Save the self relative positions')
    parser.add_argument('--low_res_init', action='store_true', default=False, help='Low-resolution init, use this when image is of high-resolution (>=2K)')

    args = parser.parse_args()
    
    # Check model format and run appropriate inference
    if is_onnx_model(args.checkpoint_path):
        print("Detected ONNX model format, using ONNX Runtime for inference...")
        run_onnx(args)
    else:
        print("Detected PyTorch model format, using PyTorch for inference...")
        run(args)

if __name__ == "__main__":
    main()