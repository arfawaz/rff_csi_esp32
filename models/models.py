#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:16:46 2025

@author: fawaz
"""
###############################################################################
#imports
import torch.nn as nn
import torch
from torchvision.models import ResNet50_Weights
from torchvision import models
from transformers import ViTConfig, ViTForImageClassification, AdamW
import math
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
import hashlib
import math
import torch.nn as nn
import torch.nn.functional as F

###############################################################################
###############################################################################
#Simple CNN model 

'''
This model is designed specifically to classify the data from Adam Plutos.This model has two Convolution Layers.
Input data is of shape 64by2. This model is mostly hardcoded and
only the number of classes (num_classes) can be varied.

'''
class SimpleCNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 1),padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1),padding=(1, 0))
        self.fc1 = nn.Linear(4096, 128)  # Adjusted based on the output shape of conv layers
        self.fc2 = nn.Linear(128, num_classes)  # 4 output classes
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(0, 0))

    def forward(self, x):
        #the shape of the input is : 999by2
        x = torch.relu(self.conv1(x)) #the shape after the first convolution lahyer is : 
        x = torch.relu(self.conv2(x))
        #x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

###############################################################################


class CNN_2(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 2, proj_dim: int = 256, l2norm: bool = True):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        self.classifier = nn.Linear(proj_dim, num_classes)
        self.l2norm = l2norm

    @staticmethod
    def _to_BCL(x: torch.Tensor) -> torch.Tensor:
        """
        Normalize CSI shapes to [B, C=2, L=64].
        Accepts:
          [B, 2, 64]           -> return as is
          [B, 64, 2]           -> permute to [B, 2, 64]
          [B, 1, 64, 2]        -> squeeze+permute to [B, 2, 64]
          [B, 64, 2, 1] / etc. -> try to move last dim=2 into channels
        """
        if x.dim() == 3:
            if x.shape[1] == 2:           # [B, 2, 64]
                return x
            if x.shape[2] == 2:           # [B, 64, 2]
                return x.permute(0, 2, 1).contiguous()
        if x.dim() == 4:
            # Common case from old 2D pipelines: [B, 1, 64, 2]
            if x.shape[1] == 1 and x.shape[3] == 2:
                x = x.squeeze(1)          # [B, 64, 2]
                return x.permute(0, 2, 1).contiguous()  # [B, 2, 64]
            # If someone left a trailing singleton: [B, 2, 64, 1]
            if x.shape[1] == 2 and x.shape[3] == 1:
                return x.squeeze(-1).contiguous()       # [B, 2, 64]
        raise ValueError(f"Unexpected CSI tensor shape {tuple(x.shape)}; need [B,2,64] logically.")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_BCL(x).float()
        h = self.feat(x).squeeze(-1)           # [B, 128]
        z = self.norm(self.proj(h))            # [B, proj_dim]
        if self.l2norm:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        z = self.encode(x)
        logits = self.classifier(z)
        return (logits, z) if return_embedding else logits


###############################################################################

# vit_model_2

'''
This model is used to do classification task on caa input data of shape 64by2
using ViT based model. We are using a built-in transformer model from huggingface
called ViTForImageClassification which takes in a ViTConfig file which contains the
details of the model like input size, number of classes, attention head etc. We 
wrap this inside the nn.module() to create the vit_model_2. In this model we configure
the ViTConfig to do tokenziation by taking each of the 64by1 columns in the whole
64by2 and embedding them. This is achieved by setting the convolutional filter
patch size as (64,1). This model is derived from the vit_model_2 in the models.py
file in the repository named "caa_authentication".
'''

class vit_model_2(nn.Module):  # Defining a custom ViT model class inheriting from nn.Module
    def __init__(self, input_dim=(64, 2), num_classes=15, hidden_size=768, 
                 num_attention_heads=12, num_hidden_layers=12, intermediate_size=3072, 
                 patch_size=(64, 1), num_channels=1):
        
        """
        Initializes the Vision Transformer (ViT) model with custom configurations.

        Args:
        - input_dim (tuple): Dimensions of the input data (height, width). Default is (64, 2).
        - num_classes (int): Number of output classes for classification. Default is 15.
        - hidden_size (int): Size of the transformer hidden layers. Default is 768.
        - num_attention_heads (int): Number of attention heads in the transformer layers. Default is 12.
        - num_hidden_layers (int): Number of transformer layers. Default is 12.
        - intermediate_size (int): Size of the intermediate feed-forward layer in the transformer. Default is 3072.
        - patch_size (tuple): Size of each patch the model processes. Default is (64, 1).
        - num_channels (int): Number of input channels. Default is 1 for grayscale data.
        """
        
        super(vit_model_2, self).__init__()  # Calls the constructor of the parent class (nn.Module)
        
        # Store the model hyperparameters
        self.input_dim = input_dim  # Input image dimensions (Height, Width)
        self.num_classes = num_classes  # Number of classification labels
        self.hidden_size = hidden_size  # Transformer hidden layer size
        self.num_attention_heads = num_attention_heads  # Number of attention heads
        self.num_hidden_layers = num_hidden_layers  # Number of transformer layers
        self.intermediate_size = intermediate_size  # Feed-forward network size
        self.patch_size = patch_size  # Patch size for dividing the input image
        self.num_channels = num_channels  # Number of channels (e.g., grayscale = 1, RGB = 3)

        # Create ViT Configuration object with the specified parameters
        self.ViTConfig = ViTConfig(
            image_size=self.input_dim,  # Specifies the input image dimensions (height, width)
            num_labels=self.num_classes,  # Number of classes in the output classification
            hidden_size=self.hidden_size,  # Size of hidden layers in the transformer
            num_attention_heads=self.num_attention_heads,  # Number of self-attention heads per transformer layer
            num_hidden_layers=self.num_hidden_layers,  # Total transformer encoder layers
            intermediate_size=self.intermediate_size,  # Size of the feed-forward layer inside each transformer block
            patch_size=self.patch_size,  # Size of image patches that will be fed to the transformer
            num_channels=self.num_channels,  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        )

        # Initialize the Vision Transformer model for image classification using the defined configuration
        self.ViTForImageClassification = ViTForImageClassification(self.ViTConfig)
        
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor representing an image or batch of images.

        Returns:
        - torch.Tensor: The output logits from the ViT classification model.
        """
        x = self.ViTForImageClassification(x)  # Pass input through the ViT model
        return x

###############################################################################

# ResNet50CSI

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision import models


class ResNet50CSI(nn.Module):
    """
    NEW: ResNet-50 adapter for CSI tensors.

    INPUT (to forward):
      x: [B, 1, 64, 2]
        - B: batch size
        - 1: single channel (e.g., magnitude or a single stacked feature)
        - 64: 'height' (e.g., subcarriers)
        - 2:  'width'  (e.g., I/Q or two features)

    ADAPTATION STEPS (inside forward):
      1) Bilinear resize to 224×224:
           [B, 1, 64, 2] → [B, 1, 224, 224]
      2) Channel repeat to 3 channels (match ImageNet pretrained stem):
           [B, 1, 224, 224] → [B, 3, 224, 224]
      3) Feed to ResNet-50 backbone (final FC replaced to num_classes):
           [B, 3, 224, 224] → logits [B, num_classes]
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # Load a torchvision ResNet-50; swap the final FC for our classes.
        # If pretrained=True, use ImageNet weights; else randomly initialize.
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_feats = self.backbone.fc.in_features  # 2048 for ResNet-50
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as [B, 1, 64, 2]
        # Step 1 — spatially upsample to the canonical 224×224 for ResNet:
        #   [B, 1, 64, 2] → [B, 1, 224, 224]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # Step 2 — duplicate the single channel across RGB:
        #   [B, 1, 224, 224] → [B, 3, 224, 224]
        x = x.repeat(1, 3, 1, 1)

        # Step 3 — ResNet-50 forward pass:
        #   [B, 3, 224, 224] → [B, num_classes] (logits)
        return self.backbone(x)


###############################################################################

# CSI_CLIP

# CSIEncoder
class CSIEncoder(nn.Module):
    
    """
   1D Conv encoder for CSI shaped [B, 2, 64] (channels = [magnitude, angle]).
   You can swap this to your complex CNN or a Transformer later.
   """
    
    def __init__(self,in_ch=2, proj_dim = 256):
        super().__init__()
        
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels = in_ch, out_channels = 64, kernel_size = 5, padding = 2), nn.ReLU(),
            nn.Conv1d(in_channels = 64, out_channels=128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> [B, 128, 1]
        )
        
        self.proj = nn.Linear(128,proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        
    
    def forward(self,x): # x: [B, 2, 64]
        h = self.feat(x).squeeze(-1) # [B, 128]
        z = self.norm(self.proj(h)) # [B, d]
        return F.normalize(z,dim = -1) # unit-norm embeddings
    

# CSIResNet50Encoder

class CSIResNet50Encoder(nn.Module):
    def __init__(self, proj_dim=256, pretrained=True, freeze_backbone_bn=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the final FC to expose the 2048-d pooled feature
        in_feats = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Project to CLIP embedding dim and layer-norm
        self.proj = nn.Linear(in_feats, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)

        # Optionally freeze BatchNorm stats if you like (helps small batches)
        if freeze_backbone_bn:
            self._freeze_bn(self.backbone)

    @staticmethod
    def _freeze_bn(module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x_2x64: torch.Tensor) -> torch.Tensor:
        """
        x_2x64: [B, 2, 64]
        -> reshape to [B, 1, 64, 2]
        -> resize to [B, 1, 224, 224]
        -> repeat channels -> [B, 3, 224, 224]
        -> ResNet-50 -> [B, 2048]
        -> proj + LN -> [B, d]
        -> L2 normalize
        """
        # Pack the two channels (e.g., mag/angle) into width=2 with a single channel
        x = x_2x64.permute(0, 2, 1).unsqueeze(1)   # [B, 64, 2] -> [B, 1, 64, 2]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)                   # [B, 3, 224, 224]

        feats = self.backbone(x)                   # [B, 2048]
        z = self.ln(self.proj(feats))              # [B, d]
        return F.normalize(z, dim=-1)
    

# LabelEmbedder

class LabelEmbedder(nn.Module):
    def __init__(self,num_classes, dim = 256):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_classes, embedding_dim=dim)
        nn.init.normal_(self.emb.weight, std=0.02)
        
    def forward(self,y):
        z = self.emb(y)
        return F.normalize(z, dim=-1)
    
    
    def table(self):
        
        return F.normalize(self.emb.weight,dim=1)
    
###############################################################################


def _mac_hex_to_vec(mac_str: str, out_dim: int = 64) -> torch.Tensor:
    """
    Deterministically maps a hex MAC string to a real vector in R^{out_dim}.
    - Uses SHA-256 of a canonicalized MAC (lower, strip colons).
    - Maps bytes -> floats in [-1, 1], then z-score normalizes.

    This function has NO learnable parameters and is deterministic across runs.
    """
    canon = mac_str.lower().replace(":", "").strip()
    h = hashlib.sha256(canon.encode("utf-8")).digest()  # 32 bytes
    # Tile or truncate to reach out_dim bytes
    raw = (h * ((out_dim + len(h) - 1) // len(h)))[:out_dim]
    v = torch.tensor(list(raw), dtype=torch.float32)
    v = v / 255.0 * 2.0 - 1.0  # map to [-1, 1]
    # z-score normalize (avoid zero std with eps)
    v = (v - v.mean()) / (v.std().clamp_min(1e-6))
    return v

class LabelHexProjector(nn.Module):
    """
    MAC-only label embedder:
      1) Precompute a deterministic vector φ_hex(mac) ∈ R^{hex_dim} for each class.
      2) Apply a learnable linear map W ∈ R^{hex_dim × D} to get 256-D embeddings.
      3) L2-normalize outputs.

    This replaces the usual nn.Embedding with a "deterministic input → learnable projector".
    """
    def __init__(self, mac_id_list, dim: int = 256, hex_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.hex_dim = hex_dim
        self.mac_id_list = list(mac_id_list)

        # Precompute base vectors as a fixed table [C, hex_dim]
        base = torch.stack([_mac_hex_to_vec(m, out_dim=hex_dim) for m in self.mac_id_list], dim=0)
        self.register_buffer("_base", base)  # not learnable

        # Learnable projection
        self.proj = nn.Linear(hex_dim, dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, y_idx: torch.Tensor) -> torch.Tensor:
        # [B, hex_dim] -> [B, D]
        v = self._base.index_select(0, y_idx)
        z = self.proj(v)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def table(self) -> torch.Tensor:
        # Class prototype table [C, D]
        z = self.proj(self._base)
        return F.normalize(z, dim=-1)

class MacLocationRFF(nn.Module):
    """
    Fixed (non-learnable) 3D-location embedding using Random Fourier Features (RFF).
    Goal: preserve distances/angles structure in a 256-D feature.

    φ_rff(x) = [cos(xΩ), sin(xΩ)] with Ω ∈ R^{3×(D/2)} sampled from N(0, σ^2).
    Everything here is REGISTERED AS BUFFERS (no gradients).

    Args:
      mac_id_list : list[str], same order as your class indices
      mac_to_xyz  : dict[str] -> (x,y,z), coordinates (e.g., meters)
      dim         : output embedding dim (use 256)
      sigma       : frequency scale (smaller => slower variation). Try 0.5..2.0
      seed        : to fix Ω deterministically
      coord_scale : multiply (x,y,z) before projection (helps unit matching)
    """
    def __init__(self, mac_id_list, mac_to_xyz: dict, dim: int = 256,
                 sigma: float = 1.0, seed: int = 20250910, coord_scale: float = 1.0):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for [cos, sin] features"
        self.dim = dim
        self.mac_id_list = list(mac_id_list)

        # Build Ω ∈ R^{3×(D/2)} deterministically
        gen = torch.Generator().manual_seed(seed)
        omega = torch.randn(3, dim // 2, generator=gen) * sigma  # N(0, σ^2)
        self.register_buffer("_omega", omega)

        # Build location table [C, D] with φ_rff(x)
        loc_rows = []
        for mac in self.mac_id_list:
            xyz = mac_to_xyz.get(mac, (0.0, 0.0, 0.0))
            x = torch.tensor(xyz, dtype=torch.float32) * coord_scale  # [3]
            proj = x @ self._omega  # [D/2]
            feat = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [D]
            # z-score then L2 normalize per row to stabilize scales
            feat = (feat - feat.mean()) / feat.std().clamp_min(1e-6)
            loc_rows.append(F.normalize(feat, dim=-1))
        table = torch.stack(loc_rows, dim=0)  # [C, D]
        self.register_buffer("_table", table)

    def forward(self, y_idx: torch.Tensor) -> torch.Tensor:
        return self._table.index_select(0, y_idx)  # already L2-normalized

    @torch.no_grad()
    def table(self) -> torch.Tensor:
        return self._table  # [C, D], L2-normalized rows

class LabelHexPlusLoc(nn.Module):
    """
    Combined label embedding:
       z_label = normalize( z_mac_hex + λ * z_loc_rff )

    - z_mac_hex comes from LabelHexProjector (learnable).
    - z_loc_rff comes from MacLocationRFF (fixed, non-learnable).
    - λ is a fixed mixing weight (float).

    Both forward(y) and table() return 256-D L2-normalized embeddings.
    """
    def __init__(self, mac_id_list, mac_to_xyz: dict,
                 dim: int = 256, hex_dim: int = 64,
                 lambda_pos: float = 1.0, sigma: float = 1.0,
                 seed: int = 20250910, coord_scale: float = 1.0):
        super().__init__()
        self.mac = LabelHexProjector(mac_id_list, dim=dim, hex_dim=hex_dim)
        self.loc = MacLocationRFF(mac_id_list, mac_to_xyz, dim=dim,
                                  sigma=sigma, seed=seed, coord_scale=coord_scale)
        self.register_buffer("_lambda_pos", torch.tensor(float(lambda_pos)))

    def forward(self, y_idx: torch.Tensor) -> torch.Tensor:
        z = self.mac(y_idx) + self._lambda_pos * self.loc(y_idx)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def table(self) -> torch.Tensor:
        z = self.mac.table() + self._lambda_pos * self.loc.table()
        return F.normalize(z, dim=-1)


###############################################################################
# clip model
    
class CSI_CLIP(nn.Module):
    def __init__(self,csi_encoder: nn.Module, label_encoder: nn.Module):
        super().__init__()
        self.csi = csi_encoder
        self.txt = label_encoder
        self.logit_scale = nn.Parameter()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))  # ~ ln(14.285)
    
    def forward(self,csi_batch, y_batch):
        
        """
        Returns CLIP logits [B, B] comparing CSI ↔ labels of the SAME batch.
        """
        
        zc = self.csi(csi_batch) # [B, d]
        zt = self.txt(y_batch) # [B, d]
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * zc @ zt.t()  # [B, B]
    
    @torch.no_grad()
    def encode_csi(self, csi_batch):
        return self.csi(csi_batch)
    
    @torch.no_grad()
    def class_table(self):
        return self.txt.table()


