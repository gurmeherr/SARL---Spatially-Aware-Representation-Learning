# ============================================================
# 1. Install dependencies
# ============================================================

!pip install -q pytorch-lightning==2.3.0 torch torchvision pandas matplotlib


# ============================================================
# 2. Imports
# ============================================================

import os
import glob
import random
from dataclasses import dataclass
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 3. Reproducibility and GPU check
# ============================================================

SEED = 42
pl.seed_everything(SEED, workers=True)

print("PyTorch:", torch.__version__)
print("Lightning:", pl.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))


# ============================================================
# 4. Config
# ============================================================

@dataclass
class SARLConfig:
    img_size: int = 224

    batch_size: int = 256
    num_workers: int = 8
    max_epochs: int = 100

    base_lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_decay: float = 0.996

    sal_w: float = 0.10
    ppda_w: float = 0.05
    ram_w: float = 0.02

    ppda_grid: int = 7
    ppda_K: int = 32
    ppda_tau: float = 0.1

    ram_grid: int = 6

    seed: int = 42


cfg = SARLConfig(
    img_size=224,
    batch_size=256,
    num_workers=8,
    max_epochs=100,
    base_lr=1e-3,
    weight_decay=1e-4,
    ema_decay=0.996,
    sal_w=0.10,
    ppda_w=0.05,
    ram_w=0.02,
    ppda_grid=7,
    ppda_K=32,
    ppda_tau=0.1,
    ram_grid=6,
    seed=42
)

pl.seed_everything(cfg.seed, workers=True)


# ============================================================
# 5. Dataset and augmentations with geometry metadata
# ============================================================

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class SARLViewTransform:
    """
    Generates one augmented view and returns:
        image tensor
        geometry metadata

    Tracked geometric transforms:
        - RandomResizedCrop
        - HorizontalFlip

    Photometric transforms:
        - ColorJitter
        - Grayscale
        - GaussianBlur
    """

    def __init__(self, size=224):
        self.size = size

        self.crop_scale = (0.2, 1.0)
        self.crop_ratio = (3.0 / 4.0, 4.0 / 3.0)

        self.hflip_p = 0.5

        self.color_jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1
        )

        self.color_jitter_p = 0.8
        self.gray_p = 0.2

        self.blur_kernel = int(0.1 * size)
        if self.blur_kernel % 2 == 0:
            self.blur_kernel += 1

        self.blur = T.GaussianBlur(
            kernel_size=self.blur_kernel,
            sigma=(0.1, 2.0)
        )

        self.normalize = T.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __call__(self, img: Image.Image):
        orig_w, orig_h = img.size

        i, j, h, w = T.RandomResizedCrop.get_params(
            img,
            scale=self.crop_scale,
            ratio=self.crop_ratio
        )

        img = TF.resized_crop(
            img,
            top=i,
            left=j,
            height=h,
            width=w,
            size=[self.size, self.size],
            interpolation=InterpolationMode.BILINEAR
        )

        do_flip = random.random() < self.hflip_p

        if do_flip:
            img = TF.hflip(img)

        if random.random() < self.color_jitter_p:
            img = self.color_jitter(img)

        if random.random() < self.gray_p:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)

        img = self.blur(img)

        img = TF.to_tensor(img)
        img = self.normalize(img)

        meta = {
            "orig_h": torch.tensor(float(orig_h), dtype=torch.float32),
            "orig_w": torch.tensor(float(orig_w), dtype=torch.float32),
            "crop_top": torch.tensor(float(i), dtype=torch.float32),
            "crop_left": torch.tensor(float(j), dtype=torch.float32),
            "crop_h": torch.tensor(float(h), dtype=torch.float32),
            "crop_w": torch.tensor(float(w), dtype=torch.float32),
            "flip": torch.tensor(float(do_flip), dtype=torch.float32),
            "view_size": torch.tensor(float(self.size), dtype=torch.float32),
        }

        return img, meta


class SARLTwoCropTransform:
    """
    Returns two independently augmented views and their metadata.
    """

    def __init__(self, size=224):
        self.transform = SARLViewTransform(size=size)

    def __call__(self, img: Image.Image):
        x1, meta1 = self.transform(img)
        x2, meta2 = self.transform(img)
        return x1, x2, meta1, meta2


class ImagePathsDataset(Dataset):
    """
    Recursively collects all images under root.

    Returns:
        x1, x2, meta1, meta2
    """

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        self.paths = sorted([
            p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if p.lower().endswith(VALID_EXTS)
        ])

        if len(self.paths) == 0:
            raise FileNotFoundError(f"No images found under: {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Could not open image: {path}") from e

        if self.transform is None:
            raise ValueError("Please provide SARLTwoCropTransform.")

        x1, x2, meta1, meta2 = self.transform(img)

        return x1, x2, meta1, meta2


# ============================================================
# 6. Dataset path and dataset creation
# ============================================================

# Change this path depending on your Colab setup.
DATA_ROOT = "/content/ViTacTip_Dataset_Final"

# For Google Drive, use something like:
# DATA_ROOT = "/content/drive/MyDrive/ViTacTip_Dataset_Final"

transform = SARLTwoCropTransform(size=cfg.img_size)

train_ds = ImagePathsDataset(
    root=DATA_ROOT,
    transform=transform
)

print("Total images found:", len(train_ds))
print("Using 100% of images for SSL training.")
print("Example image path:", train_ds.paths[0])


# ============================================================
# 7. Augmentation visualization immediately after dataset creation
# ============================================================

def unnormalize_img(x):
    """
    Convert normalized tensor from [-1, 1] back to [0, 1].
    """

    x = x.detach().cpu()
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()

    return x


x1_vis, x2_vis, meta1_vis, meta2_vis = train_ds[0]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(unnormalize_img(x1_vis))
plt.title("Augmented View 1")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(unnormalize_img(x2_vis))
plt.title("Augmented View 2")
plt.axis("off")

plt.tight_layout()
plt.show()

print("View 1 metadata:", meta1_vis)
print("View 2 metadata:", meta2_vis)


# ============================================================
# 8. Geometry correspondence utilities
# ============================================================

def _meta_to_device(meta: Dict[str, torch.Tensor], device):
    return {k: v.to(device) for k, v in meta.items()}


def make_feature_correspondence_grid(
    source_meta: Dict[str, torch.Tensor],
    target_meta: Dict[str, torch.Tensor],
    H: int,
    W: int,
    device
):
    """
    Build a grid that maps source feature coordinates to target feature coordinates.

    Used for SAL.

    For each source feature location:
        source feature coord
        -> source view pixel coord
        -> original image coord
        -> target view pixel coord
        -> target feature coord
    """

    source_meta = _meta_to_device(source_meta, device)
    target_meta = _meta_to_device(target_meta, device)

    B = source_meta["crop_top"].shape[0]

    y_base = torch.arange(H, device=device).float() + 0.5
    x_base = torch.arange(W, device=device).float() + 0.5

    yy, xx = torch.meshgrid(y_base, x_base, indexing="ij")

    yy = yy.unsqueeze(0).expand(B, H, W)
    xx = xx.unsqueeze(0).expand(B, H, W)

    source_view_size = source_meta["view_size"].view(B, 1, 1)

    y_source = yy * source_view_size / H
    x_source = xx * source_view_size / W

    source_flip = source_meta["flip"].view(B, 1, 1)

    x_source_unflipped = torch.where(
        source_flip > 0.5,
        source_view_size - x_source,
        x_source
    )

    y_orig = (
        source_meta["crop_top"].view(B, 1, 1)
        +
        y_source * source_meta["crop_h"].view(B, 1, 1) / source_view_size
    )

    x_orig = (
        source_meta["crop_left"].view(B, 1, 1)
        +
        x_source_unflipped * source_meta["crop_w"].view(B, 1, 1) / source_view_size
    )

    target_view_size = target_meta["view_size"].view(B, 1, 1)

    y_target = (
        y_orig - target_meta["crop_top"].view(B, 1, 1)
    ) * target_view_size / target_meta["crop_h"].view(B, 1, 1)

    x_target_unflipped = (
        x_orig - target_meta["crop_left"].view(B, 1, 1)
    ) * target_view_size / target_meta["crop_w"].view(B, 1, 1)

    target_flip = target_meta["flip"].view(B, 1, 1)

    x_target = torch.where(
        target_flip > 0.5,
        target_view_size - x_target_unflipped,
        x_target_unflipped
    )

    valid = (
        (x_target >= 0.0)
        & (x_target <= target_view_size - 1.0)
        & (y_target >= 0.0)
        & (y_target <= target_view_size - 1.0)
    )

    grid_x = 2.0 * (x_target / target_view_size) - 1.0
    grid_y = 2.0 * (y_target / target_view_size) - 1.0

    grid = torch.stack([grid_x, grid_y], dim=-1)
    mask = valid.float().unsqueeze(1)

    return grid, mask


def map_region_indices(
    source_meta: Dict[str, torch.Tensor],
    target_meta: Dict[str, torch.Tensor],
    S: int,
    device
):
    """
    Region-level mapping for RAM.

    Maps each source region center in an S x S grid to a target region.
    """

    source_meta = _meta_to_device(source_meta, device)
    target_meta = _meta_to_device(target_meta, device)

    B = source_meta["crop_top"].shape[0]
    P = S * S

    region_y = torch.arange(S, device=device).float()
    region_x = torch.arange(S, device=device).float()

    yy, xx = torch.meshgrid(region_y, region_x, indexing="ij")

    yy = yy.flatten().view(1, P)
    xx = xx.flatten().view(1, P)

    source_view_size = source_meta["view_size"].view(B, 1)

    y_source = (yy + 0.5) * source_view_size / S
    x_source = (xx + 0.5) * source_view_size / S

    source_flip = source_meta["flip"].view(B, 1)

    x_source_unflipped = torch.where(
        source_flip > 0.5,
        source_view_size - x_source,
        x_source
    )

    y_orig = (
        source_meta["crop_top"].view(B, 1)
        +
        y_source * source_meta["crop_h"].view(B, 1) / source_view_size
    )

    x_orig = (
        source_meta["crop_left"].view(B, 1)
        +
        x_source_unflipped * source_meta["crop_w"].view(B, 1) / source_view_size
    )

    target_view_size = target_meta["view_size"].view(B, 1)

    y_target = (
        y_orig - target_meta["crop_top"].view(B, 1)
    ) * target_view_size / target_meta["crop_h"].view(B, 1)

    x_target_unflipped = (
        x_orig - target_meta["crop_left"].view(B, 1)
    ) * target_view_size / target_meta["crop_w"].view(B, 1)

    target_flip = target_meta["flip"].view(B, 1)

    x_target = torch.where(
        target_flip > 0.5,
        target_view_size - x_target_unflipped,
        x_target_unflipped
    )

    valid = (
        (x_target >= 0.0)
        & (x_target <= target_view_size - 1.0)
        & (y_target >= 0.0)
        & (y_target <= target_view_size - 1.0)
    )

    target_rx = torch.clamp(
        (x_target / target_view_size * S).long(),
        min=0,
        max=S - 1
    )

    target_ry = torch.clamp(
        (y_target / target_view_size * S).long(),
        min=0,
        max=S - 1
    )

    mapped_indices = target_ry * S + target_rx

    return mapped_indices, valid


# ============================================================
# 9. Backbone and heads
# ============================================================

class ResNet18Backbone(nn.Module):
    """
    ResNet-18 encoder.

    Returns:
        h: final 512-dim global pooled representation
        features: [layer2, layer3, layer4]
    """

    def __init__(self):
        super().__init__()

        rn = torchvision.models.resnet18(weights=None)

        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool

        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        self.avgpool = rn.avgpool

        self.out_dim = 512

    def forward(self, x, return_feats: bool = True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        h = self.avgpool(f4).flatten(1)

        if return_feats:
            return h, [f2, f3, f4]

        return h


def mlp(in_dim, hidden_dim, out_dim, bn_last=False):
    layers = [
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, out_dim)
    ]

    if bn_last:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))

    return nn.Sequential(*layers)


class BYOLHead(nn.Module):
    """
    Online projection head g_theta and prediction head q_theta.
    """

    def __init__(
        self,
        in_dim=512,
        proj_hidden=2048,
        proj_out=256,
        pred_hidden=512
    ):
        super().__init__()

        self.projector = mlp(
            in_dim=in_dim,
            hidden_dim=proj_hidden,
            out_dim=proj_out,
            bn_last=True
        )

        self.predictor = mlp(
            in_dim=proj_out,
            hidden_dim=pred_hidden,
            out_dim=proj_out,
            bn_last=False
        )

    def forward(self, h):
        z = self.projector(h)
        p = self.predictor(z)
        return z, p


# ============================================================
# 10. Global BYOL loss
# ============================================================

def global_byol_loss(p_online, z_target):
    """
    BYOL global loss.

    Squared distance between L2-normalized online prediction
    and stop-gradient target projection.
    """

    p = F.normalize(p_online, dim=1)
    z = F.normalize(z_target.detach(), dim=1)

    return 2.0 - 2.0 * (p * z).sum(dim=1).mean()


# ============================================================
# 11. SAL: Saliency Alignment
# ============================================================

class SaliencyAlignmentLoss(nn.Module):
    """
    SAL loss.

    Logic:
        - channel-wise absolute activation sum
        - L2-normalize saliency map
        - warp target saliency into source coordinates
        - MSE over valid overlapping regions
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def saliency_map(feat):
        sal = feat.abs().sum(dim=1, keepdim=True)

        denom = sal.flatten(1).norm(
            p=2,
            dim=1,
            keepdim=True
        ).clamp_min(1e-8)

        sal = sal / denom.view(-1, 1, 1, 1)

        return sal

    def forward_single_layer(
        self,
        source_feat,
        target_feat,
        source_meta,
        target_meta
    ):
        B, C, H, W = source_feat.shape
        device = source_feat.device

        source_sal = self.saliency_map(source_feat)
        target_sal = self.saliency_map(target_feat.detach())

        if target_sal.shape[-2:] != (H, W):
            target_sal = F.interpolate(
                target_sal,
                size=(H, W),
                mode="bilinear",
                align_corners=False
            )

        grid, mask = make_feature_correspondence_grid(
            source_meta=source_meta,
            target_meta=target_meta,
            H=H,
            W=W,
            device=device
        )

        warped_target_sal = F.grid_sample(
            target_sal,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        sq_error = (source_sal - warped_target_sal) ** 2
        sq_error = sq_error * mask

        denom = mask.sum().clamp_min(1.0)

        return sq_error.sum() / denom

    def forward(
        self,
        source_feats: List[torch.Tensor],
        target_feats: List[torch.Tensor],
        source_meta,
        target_meta
    ):
        assert len(source_feats) == len(target_feats)

        losses = []

        for source_feat, target_feat in zip(source_feats, target_feats):
            losses.append(
                self.forward_single_layer(
                    source_feat=source_feat,
                    target_feat=target_feat,
                    source_meta=source_meta,
                    target_meta=target_meta
                )
            )

        return torch.stack(losses).mean()


# ============================================================
# 12. PPDA: Patch-Prototype Distribution Alignment
# ============================================================

def grid_pool(feat, S):
    """
    Feature map to patch/region tokens.

    Input:
        feat: [B, C, H, W]

    Output:
        tokens: [B, P, C]
    """

    pooled = F.adaptive_avg_pool2d(feat, (S, S))
    tokens = pooled.flatten(2).permute(0, 2, 1).contiguous()

    return tokens


class PatchPrototypeDistributionAlignmentLoss(nn.Module):
    """
    PPDA loss.

    Logic:
        - compute patch-to-prototype soft assignments
        - average over patches to get image-level prototype distribution Q(x)
        - symmetric KL between online and target distributions
    """

    def __init__(self, dim=256, K=32, grid=7, tau=0.1):
        super().__init__()

        self.K = K
        self.grid = grid
        self.tau = tau

        prototypes = torch.randn(K, dim)
        prototypes = F.normalize(prototypes, dim=1)

        self.prototypes = nn.Parameter(prototypes)

    def image_distribution(self, feat):
        tokens = grid_pool(feat, self.grid)

        tokens = F.normalize(tokens, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=1)

        logits = torch.einsum(
            "bpc,kc->bpk",
            tokens,
            prototypes
        ) / self.tau

        q_patch = F.softmax(logits, dim=-1)

        Q = q_patch.mean(dim=1)
        Q = Q / Q.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return Q

    @staticmethod
    def symmetric_kl(Q_online, Q_target):
        eps = 1e-8

        Q_online = Q_online.clamp_min(eps)
        Q_target = Q_target.clamp_min(eps)

        kl_ot = (
            Q_online * (Q_online.log() - Q_target.log())
        ).sum(dim=1).mean()

        kl_to = (
            Q_target * (Q_target.log() - Q_online.log())
        ).sum(dim=1).mean()

        return kl_ot + kl_to

    def forward(self, online_feat, target_feat):
        Q_online = self.image_distribution(online_feat)
        Q_target = self.image_distribution(target_feat.detach())

        return self.symmetric_kl(Q_online, Q_target)


# ============================================================
# 13. RAM: Region Affinity Matching
# ============================================================

class RegionAffinityMatchingLoss(nn.Module):
    """
    RAM loss.

    Logic:
        - pool feature map into S x S regions
        - compute pairwise cosine-distance affinity matrix
        - compare source affinity against geometrically mapped target affinity
    """

    def __init__(self, S=6):
        super().__init__()
        self.S = S

    @staticmethod
    def affinity(region_feats):
        region_feats = F.normalize(region_feats, dim=-1)

        sim = torch.einsum(
            "bpc,bqc->bpq",
            region_feats,
            region_feats
        )

        A = 1.0 - sim

        return A

    def forward(
        self,
        source_feat,
        target_feat,
        source_meta,
        target_meta
    ):
        device = source_feat.device
        S = self.S
        P = S * S

        source_regions = grid_pool(source_feat, S)
        target_regions = grid_pool(target_feat.detach(), S)

        A_source = self.affinity(source_regions)
        A_target = self.affinity(target_regions)

        mapped_indices, valid = map_region_indices(
            source_meta=source_meta,
            target_meta=target_meta,
            S=S,
            device=device
        )

        losses = []

        B = source_feat.shape[0]

        for b in range(B):
            valid_b = valid[b]

            if valid_b.sum() < 2:
                continue

            source_idx = torch.arange(P, device=device)[valid_b]
            target_idx = mapped_indices[b][valid_b]

            A_s = A_source[b][source_idx][:, source_idx]
            A_t = A_target[b][target_idx][:, target_idx]

            losses.append(F.mse_loss(A_s, A_t))

        if len(losses) == 0:
            return source_feat.new_tensor(0.0)

        return torch.stack(losses).mean()


# ============================================================
# 14. SARL Lightning module
# ============================================================

class SARL(pl.LightningModule):
    """
    SARL model.

    Objective:
        L = L_global + lambda_SAL * L_SAL
                     + lambda_PPDA * L_PPDA
                     + lambda_RAM * L_RAM
    """

    def __init__(self, cfg: SARLConfig):
        super().__init__()

        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg

        self.online_encoder = ResNet18Backbone()

        self.online_head = BYOLHead(
            in_dim=self.online_encoder.out_dim,
            proj_hidden=2048,
            proj_out=256,
            pred_hidden=512
        )

        self.target_encoder = ResNet18Backbone()

        self.target_projector = mlp(
            in_dim=self.online_encoder.out_dim,
            hidden_dim=2048,
            out_dim=256,
            bn_last=True
        )

        self.copy_params(self.target_encoder, self.online_encoder)
        self.copy_params(self.target_projector, self.online_head.projector)

        for p in self.target_encoder.parameters():
            p.requires_grad = False

        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.sal = SaliencyAlignmentLoss()

        self.ppda = PatchPrototypeDistributionAlignmentLoss(
            dim=256,
            K=cfg.ppda_K,
            grid=cfg.ppda_grid,
            tau=cfg.ppda_tau
        )

        self.ram = RegionAffinityMatchingLoss(
            S=cfg.ram_grid
        )

    @torch.no_grad()
    def copy_params(self, dst: nn.Module, src: nn.Module):
        for p_t, p_o in zip(dst.parameters(), src.parameters()):
            p_t.data.copy_(p_o.data)

    @torch.no_grad()
    def ema_update_target(self, m: float):
        for p_t, p_o in zip(
            self.target_encoder.parameters(),
            self.online_encoder.parameters()
        ):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

        for p_t, p_o in zip(
            self.target_projector.parameters(),
            self.online_head.projector.parameters()
        ):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    def configure_optimizers(self):
        params = (
            list(self.online_encoder.parameters())
            + list(self.online_head.parameters())
            + list(self.ppda.parameters())
        )

        optimizer = torch.optim.AdamW(
            params,
            lr=self.cfg.base_lr,
            weight_decay=self.cfg.weight_decay
        )

        return optimizer

    def forward_online(self, x):
        h, feats = self.online_encoder(x, return_feats=True)
        z, p = self.online_head(h)
        return h, feats, z, p

    @torch.no_grad()
    def forward_target(self, x):
        h, feats = self.target_encoder(x, return_feats=True)
        z = self.target_projector(h)
        return h, feats, z

    def training_step(self, batch, batch_idx):
        x1, x2, meta1, meta2 = batch

        h1_o, feats1_o, z1_o, p1_o = self.forward_online(x1)
        h2_o, feats2_o, z2_o, p2_o = self.forward_online(x2)

        h1_t, feats1_t, z1_t = self.forward_target(x1)
        h2_t, feats2_t, z2_t = self.forward_target(x2)

        loss_global = (
            global_byol_loss(p1_o, z2_t)
            +
            global_byol_loss(p2_o, z1_t)
        )

        loss_sal_12 = self.sal(
            source_feats=feats1_o,
            target_feats=feats2_t,
            source_meta=meta1,
            target_meta=meta2
        )

        loss_sal_21 = self.sal(
            source_feats=feats2_o,
            target_feats=feats1_t,
            source_meta=meta2,
            target_meta=meta1
        )

        loss_sal = 0.5 * (loss_sal_12 + loss_sal_21)

        f1_layer3_o = feats1_o[1]
        f2_layer3_o = feats2_o[1]

        f1_layer3_t = feats1_t[1]
        f2_layer3_t = feats2_t[1]

        loss_ppda_12 = self.ppda(
            online_feat=f1_layer3_o,
            target_feat=f2_layer3_t
        )

        loss_ppda_21 = self.ppda(
            online_feat=f2_layer3_o,
            target_feat=f1_layer3_t
        )

        loss_ppda = 0.5 * (loss_ppda_12 + loss_ppda_21)

        loss_ram_12 = self.ram(
            source_feat=f1_layer3_o,
            target_feat=f2_layer3_t,
            source_meta=meta1,
            target_meta=meta2
        )

        loss_ram_21 = self.ram(
            source_feat=f2_layer3_o,
            target_feat=f1_layer3_t,
            source_meta=meta2,
            target_meta=meta1
        )

        loss_ram = 0.5 * (loss_ram_12 + loss_ram_21)

        loss = (
            loss_global
            +
            self.cfg.sal_w * loss_sal
            +
            self.cfg.ppda_w * loss_ppda
            +
            self.cfg.ram_w * loss_ram
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/global", loss_global, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/SAL", loss_sal, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/PPDA", loss_ppda, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/RAM", loss_ram, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_update_target(self.cfg.ema_decay)


# ============================================================
# 15. DataLoader
# ============================================================

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True if cfg.num_workers > 0 else False,
    prefetch_factor=4 if cfg.num_workers > 0 else None
)

print("Number of training images:", len(train_ds))
print("Number of training batches per epoch:", len(train_loader))


# ============================================================
# 16. Sanity check one batch
# ============================================================

batch = next(iter(train_loader))
x1, x2, meta1, meta2 = batch

print("x1 shape:", x1.shape)
print("x2 shape:", x2.shape)
print("meta keys:", meta1.keys())
print("Example crop_top:", meta1["crop_top"][:5])
print("Example flip:", meta1["flip"][:5])


# ============================================================
# 17. Model, logger, checkpoints
# ============================================================

model = SARL(cfg)

logger = CSVLogger(
    save_dir="ssl_logs",
    name="sarl_resnet18_sal_ppda_ram"
)

ckpt_dir = os.path.join(
    "ssl_logs",
    "sarl_resnet18_sal_ppda_ram",
    "checkpoints"
)

os.makedirs(ckpt_dir, exist_ok=True)

checkpoint_best = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename="sarl-r18-epoch{epoch:03d}",
    monitor="train/loss_epoch",
    mode="min",
    save_top_k=3,
    save_last=True,
    auto_insert_metric_name=False
)

checkpoint_steps = ModelCheckpoint(
    dirpath=os.path.join(ckpt_dir, "steps"),
    filename="step{step:06d}",
    every_n_train_steps=1000,
    save_top_k=-1
)

lr_monitor = LearningRateMonitor(logging_interval="step")


# ============================================================
# 18. Trainer setup
# ============================================================

num_gpus = torch.cuda.device_count()

if num_gpus >= 2:
    accelerator = "gpu"
    devices = 2
    strategy = "ddp_notebook"
elif num_gpus == 1:
    accelerator = "gpu"
    devices = 1
    strategy = "auto"
else:
    accelerator = "cpu"
    devices = 1
    strategy = "auto"

print("Accelerator:", accelerator)
print("Devices:", devices)
print("Strategy:", strategy)

try:
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision="16-mixed" if accelerator == "gpu" else "32-true",
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=10,
        callbacks=[
            checkpoint_best,
            checkpoint_steps,
            lr_monitor
        ]
    )

except MisconfigurationException as e:
    print("Falling back because trainer configuration failed.")
    print("Original error:", e)

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy="ddp_spawn" if num_gpus >= 2 else "auto",
        precision="16-mixed" if accelerator == "gpu" else "32-true",
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=10,
        callbacks=[
            checkpoint_best,
            checkpoint_steps,
            lr_monitor
        ]
    )


# ============================================================
# 19. Train
# ============================================================

trainer.fit(
    model,
    train_dataloaders=train_loader
)

print("Logs saved to:", logger.log_dir)
print("Checkpoints saved to:", ckpt_dir)


# ============================================================
# 20. Load best checkpoint later
# ============================================================

best_ckpt = checkpoint_best.best_model_path

print("Best checkpoint:", best_ckpt)

if best_ckpt is not None and len(best_ckpt) > 0:
    loaded_model = SARL.load_from_checkpoint(
        best_ckpt,
        cfg=cfg
    )

    loaded_model.eval()

    if torch.cuda.is_available():
        loaded_model.cuda()

    print("Loaded best checkpoint successfully.")
else:
    print("No best checkpoint found yet.")
