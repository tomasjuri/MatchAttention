"""
FastMatchStereo - Configurable MatchStereo with speed/accuracy presets for embedded devices.

This module provides multiple optimization presets for running MatchStereo on
resource-constrained devices like Raspberry Pi 4.

Presets:
- default: Original configuration (best accuracy, slowest)
- balanced: Good balance of speed and accuracy
- fast: Optimized for speed with acceptable accuracy
- faster: More aggressive optimization
- ultra_fast: Maximum speed, some accuracy loss
- rpi4_optimized: Specifically tuned for Raspberry Pi 4

Usage:
    from models.fast_match_stereo import FastMatchStereo, OPTIMIZATION_PRESETS
    
    model = FastMatchStereo(args, preset='fast')
    # or
    model = FastMatchStereo(args, preset='custom', custom_config={...})
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import trunc_normal_
from models.common import UpConv
from models.convformer import convformer
from models.attention_blocks import MatchAttentionBlock
from models.cost_volume import GlobalCorrelation


# Optimization presets for different speed/accuracy tradeoffs
# 
# CRITICAL: To use pretrained weights, these parameters MUST match the original:
#   - num_heads: [4, 4, 4, 4]     (affects attention embedding dimensions)
#   - mlp_ratios: [2, 2, 2, 2]    (affects MLP hidden dimensions)  
#   - refine_win_rs: [2, 2, 1, 1] (affects cross-attention projection dimensions)
#
# The ONLY safe optimization without retraining is reducing refine_nums (iterations).
# Fewer iterations = faster, but uses only first N layers from each block.
#
OPTIMIZATION_PRESETS = {
    'default': {
        'description': 'Original configuration - best accuracy, slowest (26 iterations)',
        'refine_win_rs': [2, 2, 1, 1],  # DO NOT CHANGE - affects weight dimensions
        'refine_nums': [8, 8, 8, 2],     # Iterations: 26 total
        'num_heads': [4, 4, 4, 4],       # DO NOT CHANGE - affects weight dimensions
        'mlp_ratios': [2, 2, 2, 2],      # DO NOT CHANGE - affects weight dimensions
        'estimated_speedup': 1.0,
    },
    'balanced': {
        'description': 'Balanced - 18 iterations (30% fewer)',
        'refine_win_rs': [2, 2, 1, 1],
        'refine_nums': [6, 6, 4, 2],     # Iterations: 18 total
        'num_heads': [4, 4, 4, 4],
        'mlp_ratios': [2, 2, 2, 2],
        'estimated_speedup': 1.4,
    },
    'fast': {
        'description': 'Fast - 13 iterations (50% fewer)',
        'refine_win_rs': [2, 2, 1, 1],
        'refine_nums': [4, 4, 4, 1],     # Iterations: 13 total
        'num_heads': [4, 4, 4, 4],
        'mlp_ratios': [2, 2, 2, 2],
        'estimated_speedup': 2.0,
    },
    'faster': {
        'description': 'Faster - 7 iterations (73% fewer)',
        'refine_win_rs': [2, 2, 1, 1],
        'refine_nums': [2, 2, 2, 1],     # Iterations: 7 total
        'num_heads': [4, 4, 4, 4],
        'mlp_ratios': [2, 2, 2, 2],
        'estimated_speedup': 3.5,
    },
    'ultra_fast': {
        'description': 'Ultra fast - 4 iterations (85% fewer, accuracy may degrade)',
        'refine_win_rs': [2, 2, 1, 1],
        'refine_nums': [1, 1, 1, 1],     # Iterations: 4 total
        'num_heads': [4, 4, 4, 4],
        'mlp_ratios': [2, 2, 2, 2],
        'estimated_speedup': 6.0,
    },
    'rpi4_optimized': {
        'description': 'RPi4 optimized - 7 iterations, best for ~400x400 images',
        'refine_win_rs': [2, 2, 1, 1],
        'refine_nums': [2, 2, 2, 1],     # Iterations: 7 total
        'num_heads': [4, 4, 4, 4],
        'mlp_ratios': [2, 2, 2, 2],
        'estimated_speedup': 3.5,
    },
}


class FastMatchStereo(nn.Module):
    """
    FastMatchStereo with configurable optimization presets.
    
    This is a modified version of MatchStereo that supports runtime configuration
    of speed/accuracy tradeoffs for deployment on embedded devices.
    
    Args:
        args: Namespace with model configuration (must include 'variant')
        preset: Name of optimization preset ('default', 'fast', 'faster', etc.)
        custom_config: Optional dict to override preset values
        max_disparity: Optional maximum disparity to limit search range
    """
    
    def __init__(self, args, preset='default', custom_config=None, max_disparity=None, drop_path=0.):
        super().__init__()
        
        # Get preset configuration
        if preset not in OPTIMIZATION_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(OPTIMIZATION_PRESETS.keys())}")
        
        config = OPTIMIZATION_PRESETS[preset].copy()
        
        # Apply custom overrides if provided
        if custom_config:
            config.update(custom_config)
        
        self.preset = preset
        self.config = config
        self.max_disparity = max_disparity
        
        # Extract parameters
        refine_win_rs = config['refine_win_rs']
        refine_nums = config['refine_nums']
        num_heads = config['num_heads']
        mlp_ratios = config['mlp_ratios']
        
        self.refine_nums = refine_nums

        self.encoder = convformer(args.variant)
        self.channels = self.encoder.dims[::-1]  # resolution low to high
        self.num_heads = num_heads
        self.head_dims = [c // h for c, h in zip(self.channels, self.num_heads)]

        self.factor = 2
        self.factor_last = 2 ** (len(self.channels) - len(refine_nums) + 2)
        
        self.field_dim = 2  # 2(flow)

        self.up_decoders = nn.ModuleList()
        self.up_masks = nn.ModuleList()
        for i in range(len(self.channels)):
            if i > 0:
                self.up_decoders.append(UpConv(self.channels[i - 1], self.channels[i]))
                self.up_masks.append(
                    nn.Sequential(
                        nn.Conv2d(self.channels[i - 1], self.channels[i - 1], 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(self.channels[i - 1], (self.factor ** 2) * 9, 1, padding=0)
                    )
                )
            else:
                self.up_decoders.append(nn.Identity())
                self.up_masks.append(nn.Identity())

        self.up_masks.append(
            nn.Sequential(
                nn.Conv2d(self.channels[-1], self.channels[-1] * 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.channels[-1] * 2, (self.factor_last ** 2) * 9, 1, padding=0)
            )
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(refine_nums))]
        
        # MatchAttention blocks
        self.match_attentions = nn.ModuleList()
        for i in range(len(refine_nums)):
            self.match_attentions.append(
                MatchAttentionBlock(
                    args, self.channels[i], win_r=refine_win_rs[i],
                    num_layer=refine_nums[i], num_head=self.num_heads[i], head_dim=self.head_dims[i],
                    mlp_ratio=mlp_ratios[i], field_dim=self.field_dim,
                    dp_rates=dp_rates[sum(refine_nums[:i]):sum(refine_nums[:i + 1])]
                )
            )

        self.init_correlation_volume = GlobalCorrelation(self.channels[0])

        self.apply(self._init_weights)
        
    def get_config_summary(self):
        """Return a summary of the current configuration."""
        total_iters = sum(self.refine_nums)
        return {
            'preset': self.preset,
            'description': self.config.get('description', ''),
            'total_iterations': total_iters,
            'estimated_speedup': self.config.get('estimated_speedup', 1.0),
            'refine_nums': self.refine_nums,
            'num_heads': self.num_heads,
            'max_disparity': self.max_disparity,
        }

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def upsample_field(self, field, mask, factor):
        """Upsample field [H/factor, W/factor, D] -> [H, W, D] using convex combination"""
        B, H, W, D = field.shape
        field = field.permute(0, 3, 1, 2)
        mask = mask.view(B, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2).to(mask.dtype)
        up_flow = F.unfold(field * factor, [3, 3], padding=1)
        up_flow = up_flow.view(B, D, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2).to(mask.dtype)
        up_flow = up_flow.permute(0, 4, 2, 5, 3, 1)
        return up_flow.reshape(B, factor * H, factor * W, D).contiguous()

    def forward(self, img0, img1, stereo=True, init_flow=None):
        """
        Estimate optical flow/disparity between pair of frames.
        
        Args:
            img0: Left/reference image [B, 3, H, W] in range [0, 255]
            img1: Right/target image [B, 3, H, W] in range [0, 255]
            stereo: If True, compute stereo disparity; if False, compute optical flow
            init_flow: Optional initial flow estimate [B, H/32, W/32, 2]
            
        Returns:
            dict with keys: 'init_flow', 'init_cv', 'field_all', 'field_up', 'self_rpos'
        """
        field_all = []

        img0 = (2 * (img0 / 255.0) - 1.0).contiguous()
        img1 = (2 * (img1 / 255.0) - 1.0).contiguous()

        x = torch.cat((img0, img1), dim=0)  # cat in batch dim

        features = self.encoder(x)  # [B*2, H, W, C]
        features = features[::-1]  # reverse 1/32, 1/16, 1/8, 1/4

        for i in range(len(features)):  # 1/32, 1/16, 1/8, 1/4
            if i == 0:
                if init_flow is None:
                    init_flow, init_cv = self.init_correlation_volume(features[i], stereo=stereo)
                else:
                    init_cv = None

                field = init_flow.clone()  # [B, H, W, 2]
                self_rpos = torch.zeros_like(field)
            else:
                features[i] = self.up_decoders[i](features[i - 1], features[i])
                up_mask = self.up_masks[i](features[i - 1].permute(0, 3, 1, 2))  # [B, C, H, W]
                self_rpos = self.upsample_field(self_rpos, up_mask, self.factor)
                field = self.upsample_field(field, up_mask, self.factor)
                field_all.append({'self': field})

            features[i], self_rpos, field, fields = self.match_attentions[i](features[i], self_rpos, field, stereo=stereo)
            field_all.extend(fields)

        if self.training:
            B = field.shape[0]
            field_up = self.upsample_field(field[:B // 2], self.up_masks[-1](features[-1][:B // 2].permute(0, 3, 1, 2)), self.factor_last)
            field_up = torch.cat((field_up, field_up), dim=0)  # dummy output
        else:
            field_up = self.upsample_field(field, self.up_masks[-1](features[-1].permute(0, 3, 1, 2)), self.factor_last)

        return {
            'init_flow': init_flow,
            'init_cv': init_cv,
            'field_all': field_all,
            'field_up': field_up,
            'self_rpos': self_rpos,
        }


def create_model(args, preset='default', checkpoint_path=None, strict=False):
    """
    Factory function to create a FastMatchStereo model with optional checkpoint loading.
    
    Args:
        args: Namespace with model configuration
        preset: Optimization preset name
        checkpoint_path: Optional path to checkpoint file
        strict: If True, require exact weight matching
        
    Returns:
        FastMatchStereo model
    """
    model = FastMatchStereo(args, preset=preset)
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=strict)
        
        if missing:
            print(f"Warning: Missing keys in checkpoint: {len(missing)} keys")
            if len(missing) <= 10:
                print(f"  {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys in checkpoint: {len(unexpected)} keys")
            if len(unexpected) <= 10:
                print(f"  {unexpected}")
    
    return model


def list_presets():
    """Print available optimization presets."""
    print("\nAvailable Optimization Presets:")
    print("=" * 70)
    for name, config in OPTIMIZATION_PRESETS.items():
        iters = sum(config['refine_nums'])
        speedup = config.get('estimated_speedup', 1.0)
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Iterations:  {iters} (refine_nums: {config['refine_nums']})")
        print(f"  Num heads:   {config['num_heads']}")
        print(f"  MLP ratios:  {config['mlp_ratios']}")
        print(f"  Win radius:  {config['refine_win_rs']}")
        print(f"  Est. speedup: {speedup:.1f}x")
    print("=" * 70)


if __name__ == '__main__':
    list_presets()

