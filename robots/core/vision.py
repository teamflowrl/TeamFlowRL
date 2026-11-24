"""Frozen, real RGB perception shared by offline caching and online control."""
from pathlib import Path
import hashlib
import os
import torch
from torch import nn
from torch.nn import functional as F


class DinoVision(nn.Module):
    output_dim = 384 * 5

    def __init__(self, source, weights, device='cuda', grid=2):
        super().__init__()
        if grid not in (2, 4):
            raise ValueError(grid)
        self.grid = grid
        self.output_dim = 384 * (1 + grid * grid)
        self.backbone = torch.hub.load(str(source), 'dinov2_vits14', source='local', pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.to(device).eval()
        self.weights_sha256 = hashlib.sha256(Path(weights).read_bytes()).hexdigest()

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError('Expected genuine uint8 RGB [batch,height,width,3]')
        x = rgb.to(self.mean.device).permute(0, 3, 1, 2).float() / 255
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, (224, 224), mode='bilinear', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        f = self.backbone.forward_features(x)
        patches = f['x_norm_patchtokens'].reshape(len(x), 16, 16, 384).permute(0, 3, 1, 2)
        spatial = F.adaptive_avg_pool2d(patches, (self.grid, self.grid)).flatten(1)
        return torch.cat([f['x_norm_clstoken'], spatial], -1).float()


def actor_observations(visual, proprio, goal, time):
    """No object pose or partner proprioception enters either actor."""
    n, agents, _ = proprio.shape
    if agents != 2 or len(visual) != n or goal.shape != (n, 3):
        raise ValueError('Expected two aligned robots and a public 3-D target instruction')
    eye = torch.eye(2, device=proprio.device).expand(n, -1, -1)
    time = time.reshape(n, 1, 1).expand(-1, 2, -1)
    return torch.cat([visual[:, None].expand(-1, 2, -1), proprio,
                      goal[:, None].expand(-1, 2, -1), time, eye], -1)


# DINOv2 code and weights: set DINO_SOURCE / DINO_WEIGHTS (or pass --dino-source / --dino-weights).
DEFAULT_DINO_SOURCE = os.environ.get('DINO_SOURCE', os.path.expandvars('${CODE_ROOT}/vendor/dinov2'))
DEFAULT_DINO_WEIGHTS = os.environ.get('DINO_WEIGHTS', os.path.expandvars('${ROBOT_ROOT}/cache/dinov2_vits14_pretrain.pth'))


def add_dino_arguments(parser):
    parser.add_argument('--dino-source', type=Path, default=Path(DEFAULT_DINO_SOURCE))
    parser.add_argument('--dino-weights', type=Path, default=Path(DEFAULT_DINO_WEIGHTS))


def verify_weights(weights, expected):
    from data import sha256
    actual = sha256(weights)
    if not expected or actual != expected:
        raise ValueError('DINO weights SHA256 missing or different from dataset/checkpoint')
    return actual


def checkpoint_camera_res(cp, requested=None):
    """RGB2 provenance is mandatory; old modes do not acquire new requirements."""
    mode = cp.get('obs_mode', cp.get('params', {}).get('obs_mode'))
    if mode != 'rgb2':
        return 224
    values = [container['camera_res'] for container in
              (cp, cp.get('contract', {}), cp.get('params', {})) if 'camera_res' in container]
    if not values or values[0] not in (128, 224) or any(v != values[0] for v in values):
        raise ValueError('Missing or inconsistent RGB2 camera resolution')
    if requested is not None and requested != values[0]:
        raise ValueError('Requested camera resolution differs from RGB2 checkpoint')
    return values[0]
