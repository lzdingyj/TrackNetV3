import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DBlock(nn.Module):
    """ Conv2D + BN + ReLU """
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TrackNet, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                       # (N,   64,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   64,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,  128,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,  128,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  256,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  256,   36,    64)
        x = self.bottleneck(x)                                          # (N,  512,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  768,   72,   128)
        x = self.up_block_1(x)                                          # (N,  256,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  384,  144,   256)
        x = self.up_block_2(x)                                          # (N,  128,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,  192,  288,   512)
        x = self.up_block_3(x)                                          # (N,   64,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x


# -----------------------------------------------------------------------------
# Improved TrackNetV3 modules
# -----------------------------------------------------------------------------
class GhostConv(nn.Module):
    """GhostNet-style lightweight convolution block."""

    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, stride=1):
        super().__init__()
        primary_dim = (out_dim + 1) // 2
        cheap_dim = out_dim - primary_dim
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_dim, primary_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(primary_dim),
            nn.ReLU(inplace=True),
        )
        groups = primary_dim if cheap_dim % primary_dim == 0 else 1
        self.has_cheap_operation = cheap_dim > 0
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(primary_dim, cheap_dim, 3, 1, 1, groups=groups, bias=False),
            nn.BatchNorm2d(cheap_dim),
            nn.ReLU(inplace=True),
        ) if self.has_cheap_operation else nn.Identity()

    def forward(self, x):
        x1 = self.primary_conv(x)
        if not self.has_cheap_operation:
            return x1
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBlock(nn.Module):
    """GhostConv x 2."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = GhostConv(in_dim, out_dim)
        self.conv2 = GhostConv(out_dim, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TripleGhostBlock(nn.Module):
    """GhostConv x 3."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = GhostConv(in_dim, out_dim)
        self.conv2 = GhostConv(out_dim, out_dim)
        self.conv3 = GhostConv(out_dim, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SpatialAttention(nn.Module):
    """Spatially aggregated self-attention on bottleneck feature maps.

    The attention matrix is computed across channels after flattening spatial
    positions. This keeps the proposed attention block trainable at the default
    288x512 input size and batch sizes used by train.py.
    """

    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(x).reshape(b, 3, c, h * w).unbind(dim=1)
        attn = (q @ k.transpose(-2, -1)) * ((h * w) ** -0.5)            # (B, C, C)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)
        return self.norm(self.proj(out)) + x


class TemporalAttention(nn.Module):
    """Temporal self-attention for per-pixel feature sequences."""

    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, c).unbind(dim=2)
        attn = (q @ k.transpose(-2, -1)) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        out = self.proj(attn @ v)
        return self.norm(out + x)


class VGGTSpatioTemporal(nn.Module):
    """Apply spatial attention per frame and temporal attention across frames."""

    def __init__(self, channels=512, num_frames=8):
        super().__init__()
        if channels % num_frames != 0:
            raise ValueError(f'channels ({channels}) must be divisible by num_frames ({num_frames}).')
        self.num_frames = num_frames
        self.frame_dim = channels // num_frames
        self.spatial_attn = SpatialAttention(self.frame_dim)
        self.temporal_pos_emb = nn.Parameter(torch.randn(1, num_frames, self.frame_dim) * 0.02)
        self.temporal_attn = TemporalAttention(self.frame_dim)

    def forward(self, x):
        b, c_all, h, w = x.shape
        if c_all % self.num_frames != 0:
            raise ValueError(f'Input channels ({c_all}) must be divisible by num_frames ({self.num_frames}).')

        x = x.view(b, self.num_frames, self.frame_dim, h, w)
        x = x.reshape(b * self.num_frames, self.frame_dim, h, w)
        x = self.spatial_attn(x)
        x = x.view(b, self.num_frames, self.frame_dim, h, w)

        x_time = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, self.num_frames, self.frame_dim)
        x_time = x_time + self.temporal_pos_emb
        x_time = self.temporal_attn(x_time)
        x_time = x_time.reshape(b, h, w, self.num_frames, self.frame_dim).permute(0, 3, 4, 1, 2)
        return x_time.reshape(b, c_all, h, w)


# Backward-compatible alias for the class name in the original proposal.
VGGT_SpatioTemporal = VGGTSpatioTemporal


class BiDirectionalPyramidFusion(nn.Module):
    """Bidirectional multi-scale fusion head for decoder features."""

    def __init__(self, out_dim=3):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.conv2 = nn.Conv2d(128, 128, 1)
        self.conv3 = nn.Conv2d(64, 128, 1)

        self.top_down_1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.top_down_2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.bottom_up_1 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.bottom_up_2 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.fusion = nn.Sequential(
            nn.Conv2d(128 * 3, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat1, feat2, feat3):
        f1 = self.conv1(feat1)                                          # low resolution, 1/4 input size
        f2 = self.conv2(feat2)                                          # middle resolution, 1/2 input size
        f3 = self.conv3(feat3)                                          # high resolution, input size

        td1 = F.interpolate(f1, size=f2.shape[-2:], mode='bilinear', align_corners=False) + f2
        td1 = self.top_down_1(td1)
        td2 = F.interpolate(td1, size=f3.shape[-2:], mode='bilinear', align_corners=False) + f3
        td2 = self.top_down_2(td2)

        bu1 = F.max_pool2d(f3, 2) + f2
        bu1 = self.bottom_up_1(bu1)
        bu2 = F.max_pool2d(bu1, 2) + f1
        bu2 = self.bottom_up_2(bu2)

        bu2_up = F.interpolate(bu2, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        bu1_up = F.interpolate(bu1, size=f3.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([bu2_up, bu1_up, td2], dim=1)
        return self.sigmoid(self.fusion(fused))


class TrackNetV3Improved(nn.Module):
    """Improved TrackNetV3 with GhostConv, spatio-temporal attention and pyramid fusion."""

    def __init__(self, in_dim=24, out_dim=8, num_frames=8):
        super().__init__()
        self.num_frames = num_frames
        self.down1 = GhostBlock(in_dim, 64)
        self.down2 = GhostBlock(64, 128)
        self.down3 = TripleGhostBlock(128, 256)
        self.bottleneck = TripleGhostBlock(256, 512)

        self.spatiotemporal = VGGTSpatioTemporal(channels=512, num_frames=num_frames)

        self.up1 = TripleGhostBlock(768, 256)
        self.up2 = GhostBlock(384, 128)
        self.up3 = GhostBlock(192, 64)

        self.multi_scale_head = BiDirectionalPyramidFusion(out_dim)

    def forward(self, x):
        x1 = self.down1(x)
        x = F.max_pool2d(x1, 2)

        x2 = self.down2(x)
        x = F.max_pool2d(x2, 2)

        x3 = self.down3(x)
        x = F.max_pool2d(x3, 2)

        x = self.bottleneck(x)
        x = self.spatiotemporal(x)

        x = F.interpolate(x, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        f1 = self.up1(x)

        x = F.interpolate(f1, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        f2 = self.up2(x)

        x = F.interpolate(f2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        f3 = self.up3(x)

        return self.multi_scale_head(f1, f2, f3)


# Backward-compatible alias for the class name in the original proposal.
TrackNet_V3_Improved = TrackNetV3Improved

    
class Conv1DBlock(nn.Module):
    """ Conv1D + LeakyReLU"""
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Double1DConv(nn.Module):
    """ Conv1DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double1DConv, self).__init__()
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2)                                   # (N,   L,   3)
        x = x.permute(0, 2, 1)                                         # (N,   3,   L)
        x1 = self.down_1(x)                                            # (N,  16,   L)
        x2 = self.down_2(x1)                                           # (N,  32,   L)
        x3 = self.down_3(x2)                                           # (N,  64,   L)
        x = self.buttleneck(x3)                                        # (N,  256,  L)
        x = torch.cat([x, x3], dim=1)                                  # (N,  384,  L)
        x = self.up_1(x)                                               # (N,  128,  L)
        x = torch.cat([x, x2], dim=1)                                  # (N,  192,  L)
        x = self.up_2(x)                                               # (N,   64,  L)
        x = torch.cat([x, x1], dim=1)                                  # (N,   96,  L)
        x = self.up_3(x)                                               # (N,   32,  L)
        x = self.predictor(x)                                          # (N,   2,   L)
        x = self.sigmoid(x)                                            # (N,   2,   L)
        x = x.permute(0, 2, 1)                                         # (N,   L,   2)
        return x


if __name__ == "__main__":
    model = TrackNetV3Improved(in_dim=9, out_dim=3, num_frames=8)
    test_in = torch.randn(1, 9, 288, 512)
    test_out = model(test_in)
    print('Input shape:', test_in.shape)
    print('Output shape:', test_out.shape)
