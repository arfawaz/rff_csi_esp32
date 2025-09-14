#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 00:26:08 2025

@author: fawaz
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import csv
import torch
import random
import os
#os.environ["TORCHDYNAMO_DISABLE"] = "1"  # try to avoid importing torch._dynamo

#%%
def parse_csi_data(csi_row):
    """
    Parses a single row of CSI data into a 64x2 PyTorch tensor.
    """
    csi_values = csi_row.split()
    if len(csi_values) != 128:
        return None  # Skip invalid CSI rows
    csi_tensor = []
    for i in range(0, 128, 2):
        try:
            magnitude = float(csi_values[i])
            angle = float(csi_values[i + 1])
            csi_tensor.append([magnitude, angle])
        except ValueError:
            return None  # Skip rows with invalid numeric values
    return torch.tensor(csi_tensor)

def process_csv_fixed_id_uniform_sampling_rssi(file_path, mac_id_list, max_samples_per_mac=50000):
    """
    Processes a CSV file to extract CSI data for specific MAC addresses and assigns labels based on their order in mac_id_list.
    Instead of selecting the first max_samples_per_mac entries, this function selects uniformly from all available entries.
    """
    mac_entries = {mac: [] for mac in mac_id_list}  # Store all CSI data for each MAC
    
    # Read CSV file and collect all valid CSI entries for each MAC
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) != 2 and len(row) != 4:
                continue  # Skip invalid rows (only support 2-column or 4-column rows)
            current_mac_id, csi_row = None, None

            # Check which column contains the CSI data (second or fourth)
            if len(row) == 2:
                current_mac_id, csi_row = row
            elif len(row) == 4:
                current_mac_id, _, _, csi_row = row  # Extract CSI row from the fourth column

            if current_mac_id not in mac_id_list:
                continue  # Skip MACs not in the specified list
            
            csi_tensor = parse_csi_data(csi_row)
            if csi_tensor is not None:
                mac_entries[current_mac_id].append(csi_tensor)  # Store valid CSI tensor
    
    # Randomly select up to max_samples_per_mac for each MAC
    data = []
    labels = []
    mac_id_to_label = {mac: i for i, mac in enumerate(mac_id_list)}  # Assign labels based on order

    for mac, entries in mac_entries.items():
        sample_size = min(len(entries), max_samples_per_mac)
        sampled_entries = random.sample(entries, sample_size)  # Uniform random selection

        data.extend(sampled_entries)
        labels.extend([mac_id_to_label[mac]] * sample_size)

    if data:
        data_ = torch.stack(data)
        data_ = data_.squeeze()
        labels_ = torch.tensor(labels, dtype=torch.long)
        return data_, labels_
    else:
        return None, None  # Return None if no valid data was processed
#%% CSIEncoder
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
    

#%%

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
    
#%%
    
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

#%%
def clip_loss(logits):
    
    B = logits.size(0)
    target = torch.arange(B, device = logits.device)
    loss_i = F.cross_entropy(logits, target)
    loss_t = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i + loss_t)
    
    
#%%
    
class CSIDataset(Dataset):
    
    def __init__(self, data, labels, normalize = True, augment = False):
        super().__init__()
        assert data.ndim ==3 and data.shape[1:] == (64,2), \
            f"Expected [N,64,2], got {tuple(data.shape)}"
        
        self.x = data.permute(0,2,1).float() # [N,64,2] -> [N,2,64]
        self.y = labels.long()
        self.normalize = normalize
        self.augment = augment
    
    def __len__(self):
        return self.x.size(0)
    
    def _normalize(self,x):
        # x: [2, 64]  -> per-channel z-score across subcarriers
        mean = x.mean(dim =1, keepdim = True)
        std = x.std(dim = -1, keepdim = True).clamp_min(1e-6)
        return (x-mean)/std
    
    def _augment(self, x):
        # light noise; you can add phase jitter or subcarrier dropout later
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x
    
    def __getitem__(self,idx):
        xi, yi = self.x[idx], self.y[idx]
        if self.normalize:
            xi = self._normalize(xi)
        if self.augment:
            xi = self._augment(xi)
            
        return xi, yi
    
    # toggle for train/eval augment policy
    @property
    def train_mode(self): return getattr(self, "_train_mode", False)
    @train_mode.setter
    def train_mode(self, v): self._train_mode = bool(v)

        
#%%

@torch.no_grad()
def evaluate_zero_shot(model,loader, device):
    """
    Classification accuracy by cosine sim: z_csi vs all class embeddings.
    """
    
    model.eval()
    C = model.class_table().size(0)
    E = model.class_table().to(device) # The trained label encoder matrix of size (C,d) where C is number of classes and d is the feature dimension.
    corr, total = 0, 0
    
    for xb, yb in loader:
        xb = xb.to(device) 
        yb = yb.to(device)
        z = model.encode_csi(xb) # output of the encoder after passing csi data in xb. The output is of size (B,d) where B is the batch size and d is the feature dimension.
        logits = z @ E.t()
        pred = logits.argmax(dim = -1)
        corr += (pred==yb).sum().item()
        total += yb.numel()
    return corr/max(total,1)



def train_clip(model, train_ds, val_ds, *, epochs = 10, batch_size = 512, lr = 1e-3, wd = 1e4, num_workers = 0, device = 'cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # mark augment mode on train set only
    train_ds.train_mode = False
    val_ds.train_mode = False
    
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True,
                              drop_last = True, num_workers = num_workers, pin_memory = True)
    
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False,
                              drop_last = False, num_workers = num_workers, pin_memory = True)
    
    opt = torch.optim.AdamW([
        {"params":model.csi.parameters()},
        {"params":model.txt.parameters()},
        {"params":[model.logit_scale]},
        ], lr = lr, weight_decay = wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_val = 0.0
    
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking = True)
            yb = yb.to(device, non_blocking = True)
            
            logits = model(xb, yb)
            loss = clip_loss(logits)
            
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            total_loss +=loss.item() * xb.size(0)
            
        scheduler.step()
        avg_loss = total_loss / (len(train_loader.dataset) // train_loader.batch_size * train_loader.batch_size)
        
        val_acc = evaluate_zero_shot(model, val_loader, device)
        
        print(f"[Epoch {ep:02d}] loss={avg_loss:.4f}  val@1={val_acc*100:.2f}%  logit_scale={model.logit_scale.exp().item():.3f}")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict()}, "best_csi_clip.pth")
        
    print(f"Best val@1: {best_val*100:.2f}%")
    return model




#%%

if __name__ == "__main__":
    torch.manual_seed(42)

    # Your code above should already have:
    #   - file_path = input("...")
    #   - data, labels = process_csv_fixed_id_uniform_sampling_rssi(...)
    # If not yet executed, you can uncomment and reuse here.
    file_path = input("Please enter the file path to the CSV file: ")
    data, labels = process_csv_fixed_id_uniform_sampling_rssi(
        file_path=file_path,
        mac_id_list= [
         "00:FC:BA:38:4B:00", \
         "00:FC:BA:38:4B:01", \
         "00:FC:BA:38:4B:02", \
         "6C:B2:AE:39:1A:A0", \
         "6C:B2:AE:39:1A:A1", \
         "70:0F:6A:DE:EC:A0", \
         "70:0F:6A:DE:EC:A1", \
         "70:0F:6A:DE:EC:A2", \
         #"00:FC:BA:27:63:00", \
         #"00:FC:BA:27:63:01", \
         #"00:FC:BA:27:63:02"
         ],
        
       
        max_samples_per_mac=40000
    )
    assert data is not None and labels is not None, "No data loaded from CSV."

    # Build dataset
    full_ds = CSIDataset(data, labels, normalize=True, augment=True)
    N = len(full_ds)
    n_train = int(0.8 * N)
    n_val   = N - n_train
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(123))

    num_classes = int(labels.max().item() + 1)
    print(f"Samples: {N} | Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = CSI_CLIP(
        csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),
        label_encoder=LabelEmbedder(num_classes=num_classes, dim=256),
    )

    # Train
    trained = train_clip(
        model, train_ds, val_ds,
        epochs=10, batch_size=512, lr=1e-3, wd=1e-4, num_workers=0, device="cuda"
    )

    # Final eval on the validation split (zero-shot style classification)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    val_acc = evaluate_zero_shot(trained, val_loader, device="cuda")
    print(f"Validation top-1 accuracy: {val_acc*100:.2f}%")


#%%

# ===========================
# Load and evaluate on new TEST file
# ===========================

# Reuse the exact class order you trained with
MAC_ID_LIST =  [
  "00:FC:BA:38:4B:00", \
  "00:FC:BA:38:4B:01", \
  "00:FC:BA:38:4B:02", \
  "6C:B2:AE:39:1A:A0", \
  "6C:B2:AE:39:1A:A1", \
  "70:0F:6A:DE:EC:A0", \
  "70:0F:6A:DE:EC:A1", \
  "70:0F:6A:DE:EC:A2", \
  #"00:FC:BA:27:63:00", \
  #"00:FC:BA:27:63:01", \
  #"00:FC:BA:27:63:02"
  ]

# 1) Load test CSV
test_file_path = input("Enter path to TEST CSV file: ")
test_data, test_labels = process_csv_fixed_id_uniform_sampling_rssi(
    file_path=test_file_path,
    mac_id_list=MAC_ID_LIST,
    max_samples_per_mac=1_000_000,  # take all available
)
assert test_data is not None and test_labels is not None, "No valid test data found."

# 2) Wrap as dataset/loader (same normalization, no augmentation)
test_ds = CSIDataset(test_data, test_labels, normalize=True, augment=False)
test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
print(f"Test samples: {len(test_ds)} | Classes: {len(MAC_ID_LIST)}")

# 3) Use the trained model in memory (variable `trained`)...
device = "cuda" if torch.cuda.is_available() else "cpu"
if 'trained' in globals():
    test_acc = evaluate_zero_shot(trained, test_loader, device=device)
    print(f"TEST top-1 accuracy (from in-memory model): {test_acc*100:.2f}%")
else:
    # ...or rebuild the model and load the saved checkpoint
    model_test = CSI_CLIP(
        csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),
        label_encoder=LabelEmbedder(num_classes=len(MAC_ID_LIST), dim=256),
    )
    ckpt = torch.load("best_csi_clip.pth", map_location=device)
    model_test.load_state_dict(ckpt["model"])
    model_test.to(device).eval()

    test_acc = evaluate_zero_shot(model_test, test_loader, device=device)
    print(f"TEST top-1 accuracy (from checkpoint): {test_acc*100:.2f}%")





#%%

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
        super(SimpleCNN, self).__init__()
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






#%% to be copied ot main from here forth

#%% testing

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import csv
import torch
import random
import os

from csi_dataset_creator_fixed_id import process_csv_fixed_id
from csi_dataset_creator_fixed_id_uniform_sampling import process_csv_fixed_id_uniform_sampling
from process_csv_fixed_id_uniform_smapling_rssi import process_csv_fixed_id_uniform_sampling_rssi
from csi_dataset_creator import process_csv
from mean_norm import mean_norm
from train_test import train, test
from train_test_loader import train_test_loader
from train_vit_model_2 import train_vit_model_2
from test_vit_model_2 import test_vit_model_2
from CustomDataset_vit_model_2 import CustomDataset_vit_model_2
from models import SimpleCNN, vit_model_2, ResNet50CSI
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW
import torch.nn as nn
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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

# clip loss
def clip_loss(logits):
    
    B = logits.size(0)
    target = torch.arange(B, device = logits.device)
    loss_i = F.cross_entropy(logits, target)
    loss_t = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i + loss_t)
    
    
# CLIP normalization/augmentation wrapper
class CSIDataset(torch.utils.data.Dataset):
    def __init__(self, base_x_2x64, base_y, normalize=True, augment=False):
        self.x = base_x_2x64
        self.y = base_y
        self.normalize = normalize
        self.augment = augment
        self._train_mode = False
    def __len__(self): return self.x.size(0)
    def _normalize(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std
    def _augment(self, x):
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x
    @property
    def train_mode(self): return self._train_mode
    @train_mode.setter
    def train_mode(self, v: bool): self._train_mode = bool(v)
    def __getitem__(self, i):
        xi, yi = self.x[i], self.y[i]
        if self.normalize: xi = self._normalize(xi)
        if self.augment and self._train_mode: xi = self._augment(xi)
        return xi, yi
    

@torch.no_grad()
def evaluate_zero_shot(model,loader, device):
    """
    Classification accuracy by cosine sim: z_csi vs all class embeddings.
    """
    
    model.eval()
    C = model.class_table().size(0)
    E = model.class_table().to(device) # The trained label encoder matrix of size (C,d) where C is number of classes and d is the feature dimension.
    corr, total = 0, 0
    
    for xb, yb in loader:
        xb = xb.to(device) 
        yb = yb.to(device)
        z = model.encode_csi(xb) # output of the encoder after passing csi data in xb. The output is of size (B,d) where B is the batch size and d is the feature dimension.
        logits = z @ E.t()
        pred = logits.argmax(dim = -1)
        corr += (pred==yb).sum().item()
        total += yb.numel()
    return corr/max(total,1)



def train_clip(model, train_ds, val_ds, *, epochs = 10, batch_size = 64, lr = 1e-3, wd = 1e4, num_workers = 0, device = 'cuda'):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # mark augment mode on train set only
    train_ds.train_mode = False
    val_ds.train_mode = False
    
    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True,
                              drop_last = True, num_workers = num_workers, pin_memory = True)
    
    val_loader = DataLoader(val_ds, batch_size = batch_size, shuffle = False,
                              drop_last = False, num_workers = num_workers, pin_memory = True)
    
    opt = torch.optim.AdamW([
        {"params":model.csi.parameters()},
        {"params":model.txt.parameters()},
        {"params":[model.logit_scale]},
        ], lr = lr, weight_decay = wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    best_val = 0.0
    
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking = True)
            yb = yb.to(device, non_blocking = True)
            
            logits = model(xb, yb)
            loss = clip_loss(logits)
            
            opt.zero_grad(set_to_none = True)
            loss.backward()
            opt.step()
            total_loss +=loss.item() * xb.size(0)
            
        scheduler.step()
        avg_loss = total_loss / (len(train_loader.dataset) // train_loader.batch_size * train_loader.batch_size)
        
        val_acc = evaluate_zero_shot(model, val_loader, device)
        
        print(f"[Epoch {ep:02d}] loss={avg_loss:.4f}  val@1={val_acc*100:.2f}%  logit_scale={model.logit_scale.exp().item():.3f}")
        
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict()}, "best_csi_clip.pth")
        
    print(f"Best val@1: {best_val*100:.2f}%")
    return model

print("done clip model cell")

#%% #%% Loading data and labels for CNN and vit_model_2

# Prompt the user for the file path
file_path = input("Please enter the file path to the CSV file: ")

data, labels = process_csv_fixed_id_uniform_sampling_rssi(file_path = file_path , mac_id_list = [
"00:FC:BA:38:4B:00", \
"00:FC:BA:38:4B:01", \
"00:FC:BA:38:4B:02", \
"6C:B2:AE:39:1A:A0", \
"6C:B2:AE:39:1A:A1", \
"70:0F:6A:DE:EC:A0", \
"70:0F:6A:DE:EC:A1", \
"70:0F:6A:DE:EC:A2", \
#"00:FC:BA:27:63:00", \
#"00:FC:BA:27:63:01", \
#"00:FC:BA:27:63:02"
], max_samples_per_mac=100000000)
    
print("Done data loading")

#%%
# ============================================================
# Shared TRAIN/VAL (from TRAIN CSV) + shared TEST (from TEST CSV)
# Reproducible & identical for CNN and CLIP
# ============================================================
import os, random
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

SEED = 20250910
SHARED_SPLIT_FILE = f"shared_train_val_split_seed{SEED}.pt"

def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_global_seed(SEED)

# ---- SAME MAC list for train & test so label ids match ----
MAC_ID_LIST = [
    "00:FC:BA:38:4B:00",
    "00:FC:BA:38:4B:01",
    "00:FC:BA:38:4B:02",
    "6C:B2:AE:39:1A:A0",
    "6C:B2:AE:39:1A:A1",
    "70:0F:6A:DE:EC:A0",
    "70:0F:6A:DE:EC:A1",
    "70:0F:6A:DE:EC:A2",
]

# --------------------------
# Build full TRAIN sets (once)
#   data:   [N, 64, 2]
#   labels: [N]
# --------------------------
labels = labels.long()
N = data.size(0)

# CNN view: [N,1,64,2] + your mean_norm()
x_cnn_full = mean_norm(data.unsqueeze(1).float())
cnn_full_ds = TensorDataset(x_cnn_full, labels)

# CLIP base view: [N,2,64] (normalize/augment inside wrapper)
x_clip_base = data.permute(0, 2, 1).float()  # [N,2,64]

class CSIDataset(torch.utils.data.Dataset):
    def __init__(self, x_2x64, y, normalize=True, augment=False):
        self.x = x_2x64
        self.y = y.long()
        self.normalize = normalize
        self.augment = augment
        self._train_mode = False
    def __len__(self): return self.x.size(0)
    def _normalize(self, x):
        m, s = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - m) / s
    def _augment(self, x):
        if torch.rand(1).item() < 0.5:
            x = x + 0.01 * torch.randn_like(x)
        return x
    @property
    def train_mode(self): return self._train_mode
    @train_mode.setter
    def train_mode(self, v): self._train_mode = bool(v)
    def __getitem__(self, i):
        xi, yi = self.x[i], self.y[i]
        if self.normalize: xi = self._normalize(xi)
        if self.augment and self._train_mode: xi = self._augment(xi)
        return xi, yi

# --------------------------
# Deterministic TRAIN/VAL split (no test from TRAIN CSV)
# --------------------------
def stratified_train_val_indices(y_tensor, train_ratio=0.9, seed=SEED):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for idx, yy in enumerate(y_tensor.tolist()):
        by_class[int(yy)].append(idx)
    train_idx, val_idx = [], []
    for c, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train_ratio))
        # keep at least 1 for val if class has >1 sample
        if n > 1:
            n_train = min(max(n_train, 1), n - 1)
        train_idx += idxs[:n_train]
        val_idx   += idxs[n_train:]
    train_idx.sort(); val_idx.sort()
    return train_idx, val_idx

if os.path.exists(SHARED_SPLIT_FILE):
    split = torch.load(SHARED_SPLIT_FILE)
    train_idx, val_idx = split["train_idx"], split["val_idx"]
    assert max(train_idx + val_idx) < N, \
        "Saved split indices exceed current TRAIN dataset size. Delete split file and re-run."
else:
    train_idx, val_idx = stratified_train_val_indices(labels, train_ratio=0.9, seed=SEED)
    torch.save({"train_idx": train_idx, "val_idx": val_idx}, SHARED_SPLIT_FILE)

print(f"[TRAIN CSV] TRAIN={len(train_idx)} | VAL={len(val_idx)}  (seed={SEED})")

# --------------------------
# Build TRAIN/VAL for BOTH models using same indices
# --------------------------
def make_loaders_for_dataset(dataset, tr_idx, va_idx, batch_size=64, num_workers=0):
    ds_train = Subset(dataset, tr_idx)
    ds_val   = Subset(dataset, va_idx)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  drop_last=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# CNN loaders (use these in your CNN train/test functions)
cnn_train_loader, cnn_val_loader = make_loaders_for_dataset(cnn_full_ds, train_idx, val_idx, batch_size=64)

# CLIP datasets (train uses aug; val no aug)
clip_train_full = CSIDataset(x_clip_base, labels, normalize=True, augment=True);  clip_train_full.train_mode = True
clip_val_full   = CSIDataset(x_clip_base, labels, normalize=True, augment=False); clip_val_full.train_mode   = False
clip_train_ds = Subset(clip_train_full, train_idx)
clip_val_ds   = Subset(clip_val_full,   val_idx)

print("[TRAIN/VAL] Shared splits ready for CNN and CLIP.")

# --------------------------
# TEST set: load from a NEW CSV ONE TIME and reuse for BOTH
# --------------------------
# Ensure deterministic sub-sampling inside your CSV loader
random.seed(SEED)

test_file_path = input("Please enter the file path to the TEST CSV file: ")

test_data, test_labels = process_csv_fixed_id_uniform_sampling_rssi(
    file_path=test_file_path,
    mac_id_list=MAC_ID_LIST,
    max_samples_per_mac=1_000_000
)
assert test_data is not None and test_labels is not None, "No valid test data found in TEST CSV."
test_labels = test_labels.long()

# CNN TEST loader (entire TEST set)
x_cnn_test = mean_norm(test_data.unsqueeze(1).float())
cnn_test_ds = TensorDataset(x_cnn_test, test_labels)
cnn_test_loader = DataLoader(cnn_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

# CLIP TEST loader (entire TEST set; normalized, no aug)
x_clip_test = test_data.permute(0, 2, 1).float()  # [M,2,64]
clip_test_ds = CSIDataset(x_clip_test, test_labels, normalize=True, augment=False)
clip_test_ds.train_mode = False
clip_test_loader = DataLoader(clip_test_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

print(f"[TEST CSV] TEST={len(test_labels)} (shared for both models)")


#%% cnn training and testing on  external dataset

num_classes = 8
learning_rate = 0.001
num_epochs = 10

# Model setup
model_1 = SimpleCNN(num_classes)
model_1 = model_1.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_1.parameters(), lr=learning_rate)

# Model Training
model_1.train()
train(model=model_1, 
      train_loader=cnn_train_loader, 
      test_loader=cnn_val_loader, 
      criterion=criterion, 
      optimizer=optimizer, 
      num_epochs=num_epochs)

# Model Testing
model_1.eval()
_ = test(model_1, cnn_test_loader)



#%% testing on clip model

# Assuming you have CSI_CLIP, CSIEncoder, LabelEmbedder, evaluate_zero_shot, and train(...) from earlier
num_classes = len(MAC_ID_LIST)
clip_model = CSI_CLIP(
    csi_encoder=CSIEncoder(in_ch=2, proj_dim=256),
    label_encoder=LabelEmbedder(num_classes=num_classes, dim=256),
)

trained_clip = train_clip(
    clip_model, clip_train_ds, clip_val_ds,
    epochs=10, batch_size=64, lr=1e-3, wd=1e-4, num_workers=0, device="cuda"
)

# Final test on the external TEST CSV (same samples as CNN)
clip_test_acc = evaluate_zero_shot(trained_clip, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[CLIP] TEST top-1: {clip_test_acc*100:.2f}%")

#%% testing on resnet50

model_3 = ResNet50CSI(num_classes=num_classes, pretrained=True).to(device)
resnet_lr = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_3.parameters(), lr=resnet_lr)

# Train
model_3.train()
train(
    model=model_3,
    train_loader=cnn_train_loader,
    test_loader=cnn_val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=4
)

# Evaluate on primary test set
model_3.eval()
_ = test(model_3, cnn_test_loader)


#%%

# ================================
# ResNet-50 CSI Encoder for CLIP
# Input expected: x ∈ R[B, 2, 64]
#   (e.g., [magnitude, angle] across 64 subcarriers)
# Internally reshaped to [B, 1, 64, 2] -> resized to 224x224 -> 3ch
# Output: L2-normalized embedding ∈ R[B, proj_dim]
# ================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

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

#%%

num_classes = len(MAC_ID_LIST)

clip_model_resnet = CSI_CLIP(
    csi_encoder=CSIResNet50Encoder(proj_dim=256, pretrained=True, freeze_backbone_bn=False),
    label_encoder=LabelEmbedder(num_classes=num_classes, dim=256),
).to(device)

trained_clip = train_clip(
    clip_model_resnet, clip_train_ds, clip_val_ds,
    epochs=4, batch_size=64, lr=1e-4, wd=1e-4, num_workers=0, device="cuda"
)

# Final test on the external TEST CSV (same samples as CNN)
clip_test_acc = evaluate_zero_shot(trained_clip, clip_test_loader, device=("cuda" if torch.cuda.is_available() else "cpu"))
print(f"[CLIP] TEST top-1: {clip_test_acc*100:.2f}%")       







    
    
    
    
    
    
    