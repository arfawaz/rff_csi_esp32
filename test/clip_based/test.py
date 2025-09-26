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



############################################################################################################################
#%% clip with position embedding 

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
from models import SimpleCNN, vit_model_2, ResNet50CSI, CSIEncoder, LabelEmbedder, CSIResNet50Encoder, CSI_CLIP
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW
import torch.nn as nn
import torch
import torch.optim as optim
from losses import clip_loss
from clip_dataset import CSIDataset
from evaluate_zero_shot import evaluate_zero_shot
from train_test import train_clip
from stratified_train_val_indices import stratified_train_val_indices
from make_loaders_for_dataset import make_loaders_for_dataset
import os, random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from set_global_seed import set_global_seed
from losses import clip_loss
from evaluate_zero_shot import evaluate_zero_shot
import os, random, torch
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset, Subset
from process_csv_fixed_id_uniform_sampling_rssi_pos import process_csv_fixed_id_uniform_sampling_rssi_pos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
from models import SimpleCNN, vit_model_2, ResNet50CSI, CSIEncoder, LabelEmbedder, CSIResNet50Encoder, CSI_CLIP
from torch.utils.data import Subset, DataLoader, TensorDataset, random_split
from transformers import AdamW
import torch.nn as nn
import torch
import torch.optim as optim
from losses import clip_loss
from clip_dataset import CSIDataset
from evaluate_zero_shot import evaluate_zero_shot
from train_test import train_clip
from stratified_train_val_indices import stratified_train_val_indices
from make_loaders_for_dataset import make_loaders_for_dataset
import os, random
from collections import defaultdict
import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from set_global_seed import set_global_seed
from losses import clip_loss
from evaluate_zero_shot import evaluate_zero_shot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#%%

# ================================================================
# CLIP for CSI with Position Embeddings (MAC + POS on label side)
# - Uses existing combined CSVs WITH POS column for train and test
# - Loads (data, mac_labels, pos_labels) with uniform per-MAC sampling
# - Shared MAC-stratified train/val split (test is separate CSV)
# - CSI encoder: ResNet-50 adapted for CSI (REUSED)
# - Label encoder: MAC + POS (learnable weight), MAC-only evaluation
# - Stochastic position drop during training
# ================================================================

import os, csv, math, random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import models
from torchvision.models import ResNet50_Weights

# --------------------
# USER SETTINGS
# --------------------
SEED = 20250910

# Point these to your EXISTING combined CSVs (no headers):
#   Columns: [MAC, RSSI, NOISE, CSI, POS]  OR  [MAC, CSI, POS]
TRAIN_CSV = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\train\train_with_pos.csv"
TEST_CSV  = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\test_with_pos.csv"

# Fixed MAC class list (order defines class indices)
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
POS_TAG_TO_ID = {f"p{i}": i-1 for i in range(1, 9)}  # p1->0 ... p8->7

# Train config
BATCH_TRAIN = 64
BATCH_EVAL  = 64
EPOCHS      = 10
LR          = 1e-3
WEIGHT_DECAY= 1e-4
P_USE_POS   = 0.5  # probability to use POS each training step

# --------------------
# Reproducibility
# --------------------
def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_global_seed(SEED)

# ================================================================
# Loader -> (data [N,64,2], mac_labels [N], pos_labels [N])
# (Mirrors your process_csv_fixed_id_uniform_sampling_rssi, now with POS)
# ================================================================
def parse_csi_data(csi_row: str):
    vals = csi_row.split()
    if len(vals) != 128:
        return None
    out = []
    for i in range(0, 128, 2):
        try:
            mag = float(vals[i]); ang = float(vals[i+1])
        except ValueError:
            return None
        out.append([mag, ang])
    return torch.tensor(out)  # [64,2]

def process_csv_fixed_id_uniform_sampling_rssi_pos(
    file_path: str,
    mac_id_list: list,
    pos_tag_to_id: dict = None,
    max_samples_per_mac: int = 50_000,
    stratify_by_pos: bool = False,
    seed: int = SEED,
):
    """
    Reads CSV with 3 or 5 columns:
      - 3: [MAC, CSI, POS]
      - 5: [MAC, RSSI, NOISE, CSI, POS]
    Returns:
      data:       [N, 64, 2] float32
      mac_labels: [N] long (0..C-1)
      pos_labels: [N] long (0..K-1)
    """
    random.seed(seed)
    mac_to_label = {m: i for i, m in enumerate(mac_id_list)}
    if pos_tag_to_id is None:
        pos_tag_to_id = {f"p{i}": i-1 for i in range(1, 9)}

    store = defaultdict(list)  # key: mac_id or (mac_id,pos_id)
    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) not in (3, 5):
                continue
            if len(row) == 3:
                mac, csi_row, pos = row
            else:
                mac, _, _, csi_row, pos = row

            if mac not in mac_to_label:
                continue

            pos_lc = str(pos).strip().lower()
            if pos_lc not in pos_tag_to_id:
                pl = pos_lc.lstrip("p")
                if pl.isdigit():
                    pos_lc = f"p{int(pl)}"
                if pos_lc not in pos_tag_to_id:
                    continue

            csi_t = parse_csi_data(csi_row)
            if csi_t is None:
                continue

            mac_id = mac_to_label[mac]
            pos_id = pos_tag_to_id[pos_lc]
            key = (mac_id, pos_id) if stratify_by_pos else mac_id
            store[key].append((csi_t, mac_id, pos_id))

    data, mac_labels, pos_labels = [], [], []
    if stratify_by_pos:
        by_mac_total = defaultdict(int)
        for (mac_id, pos_id), entries in store.items():
            budget = max_samples_per_mac - by_mac_total[mac_id]
            if budget <= 0: continue
            take = min(len(entries), budget)
            for (csi_t, m, p) in random.sample(entries, take):
                data.append(csi_t); mac_labels.append(m); pos_labels.append(p)
                by_mac_total[mac_id] += 1
    else:
        for mac_id, entries in store.items():
            take = min(len(entries), max_samples_per_mac)
            for (csi_t, m, p) in random.sample(entries, take):
                data.append(csi_t); mac_labels.append(m); pos_labels.append(p)

    if not data:
        return None, None, None

    data = torch.stack(data).squeeze().float()               # [N,64,2]
    mac_labels = torch.tensor(mac_labels, dtype=torch.long)  # [N]
    pos_labels = torch.tensor(pos_labels, dtype=torch.long)  # [N]
    return data, mac_labels, pos_labels

# ================================================================
# Shared split utils (REUSED from no-position version)
# ================================================================
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
        n_train = min(max(n_train, 1 if n > 1 else n), n-1 if n > 1 else n)
        train_idx += idxs[:n_train]
        val_idx   += idxs[n_train:]
    train_idx.sort(); val_idx.sort()
    return train_idx, val_idx

def make_loaders_for_dataset(dataset, tr_idx, va_idx, batch_size=64, num_workers=0):
    ds_train = Subset(dataset, tr_idx)
    ds_val   = Subset(dataset, va_idx)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ================================================================
# CLIP dataset (3-tuple: x, mac, pos)
# ================================================================
class CLIPDataset3(torch.utils.data.Dataset):
    """Returns (x[2,64], mac, pos)."""
    def __init__(self, x_64x2, y_mac, y_pos, normalize=True, augment=False):
        self.x = x_64x2.permute(0,2,1).float()  # [N,2,64]
        self.y_mac = y_mac.long()
        self.y_pos = y_pos.long()
        self.normalize, self.augment = normalize, augment
        self._train = False
    def __len__(self): return self.x.size(0)
    def _norm(self, x):
        m = x.mean(dim=-1, keepdim=True); s = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - m) / s
    def _aug(self, x):
        return x + 0.01*torch.randn_like(x) if torch.rand(()) < 0.5 else x
    @property
    def train_mode(self): return self._train
    @train_mode.setter
    def train_mode(self, v): self._train = bool(v)
    def __getitem__(self, i):
        xi = self._norm(self.x[i]) if self.normalize else self.x[i]
        if self.augment and self._train: xi = self._aug(xi)
        return xi, self.y_mac[i], self.y_pos[i]

# ================================================================
# Encoders & CLIP model
# ================================================================
class CSIResNet50Encoder(nn.Module):
    """
    ResNet-50 backbone -> 2048-d -> projection -> LayerNorm -> L2-normalized
    Input expected in forward: x [B,2,64]; internally reshaped to [B,1,64,2]
    (REUSED from prior task, unchanged logic)
    """
    def __init__(self, proj_dim=256, pretrained=True, freeze_backbone_bn=False):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        in_feats = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_feats, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
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
        x = x_2x64.permute(0, 2, 1).unsqueeze(1)     # [B,1,64,2]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)                     # [B,3,224,224]
        feats = self.backbone(x)                     # [B,2048]
        z = self.ln(self.proj(feats))                # [B,d]
        return F.normalize(z, dim=-1)

class CSICNNEncoder(nn.Module):
    """
    Lightweight 1D-CNN CSI encoder for inputs x [B,2,64].
    Produces a CLIP-style embedding: proj -> LayerNorm -> L2-normalize.
    """
    def __init__(self, proj_dim=256, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # -> [B,128,1]
        )
        self.proj = nn.Linear(128, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
    def forward(self, x_2x64: torch.Tensor) -> torch.Tensor:
        h = self.net(x_2x64).squeeze(-1)   # [B,128]
        z = self.ln(self.proj(h))          # [B,d]
        return F.normalize(z, dim=-1)

class LabelEmbedder(nn.Module):
    """(REUSED) MAC-only label embedder used inside MacPosEncoder."""
    def __init__(self, num_classes, dim=256):
        super().__init__()
        self.emb = nn.Embedding(num_classes, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, y):
        return F.normalize(self.emb(y), dim=-1)
    def table(self):
        return F.normalize(self.emb.weight, dim=-1)

class PositionEmbedder(nn.Module):
    def __init__(self, num_pos, dim=256):
        super().__init__()
        self.emb = nn.Embedding(num_pos, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, p):
        return F.normalize(self.emb(p), dim=-1)
    def table(self):
        return F.normalize(self.emb.weight, dim=-1)

class MacPosEncoder(nn.Module):
    """
    Text/label side for CLIP:
      z_text = normalize( z_mac + w * z_pos ) with learnable/fixed w.
    """
    def __init__(self, num_classes, num_pos, dim=256, learnable_w=True, init_w=0.5):
        super().__init__()
        self.mac = LabelEmbedder(num_classes, dim)
        self.pos = PositionEmbedder(num_pos, dim)
        if learnable_w:
            self._logit_w = nn.Parameter(torch.tensor(math.log(init_w/(1-init_w))))
        else:
            self.register_buffer("_w", torch.tensor(float(init_w)))
    def weight(self):
        if hasattr(self, "_logit_w"):
            return torch.sigmoid(self._logit_w)
        return self._w
    def forward(self, mac_y, pos_y=None, use_pos=True):
        z = self.mac(mac_y)
        if use_pos and (pos_y is not None):
            w = self.weight()
            z = F.normalize(z + w * self.pos(pos_y), dim=-1)
        return z
    # Tables for evaluation
    def class_table_mac_only(self):
        return self.mac.table()
    def class_table_avg_over_pos(self):
        Emac = self.mac.table()  # [C,d]
        Epos = self.pos.table()  # [K,d]
        w = self.weight()
        M = F.normalize(Emac[:, None, :] + w * Epos[None, :, :], dim=-1)  # [C,K,d]
        return F.normalize(M.mean(dim=1), dim=-1)  # [C,d]

class CSI_CLIP_POS(nn.Module):
    def __init__(self, csi_encoder: nn.Module, label_encoder: nn.Module):
        super().__init__()
        self.csi = csi_encoder
        self.txt = label_encoder
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))
    def forward(self, csi_batch, mac_y, pos_y=None, use_pos=True):
        zc = self.csi(csi_batch)                       # [B,d]
        zt = self.txt(mac_y, pos_y, use_pos=use_pos)   # [B,d]
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * zc @ zt.t()                     # [B,B]
    @torch.no_grad()
    def encode_csi(self, csi_batch):
        return self.csi(csi_batch)
    @torch.no_grad()
    def class_table(self, mode="mac_only"):
        if mode == "mac_only":
            return self.txt.class_table_mac_only()
        elif mode == "avg_over_pos":
            return self.txt.class_table_avg_over_pos()
        else:
            raise ValueError("mode ∈ {'mac_only','avg_over_pos'}")

# ================================================================
# Loss & Train/Eval (REUSED structure from no-position CLIP)
# ================================================================
def clip_loss_pos(logits):
    """(REUSED) symmetric InfoNCE with diagonal targets."""
    B = logits.size(0)
    target = torch.arange(B, device=logits.device)
    loss_i = F.cross_entropy(logits, target)
    loss_t = F.cross_entropy(logits.t(), target)
    return 0.5 * (loss_i + loss_t)

@torch.no_grad()
def eval_clip_mac_only(model, loader, device="cuda", mode="mac_only"):
    """(REUSED idea) z_csi vs class table, argmax over MAC classes."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    E = model.class_table(mode=mode).to(device)   # [C,d]
    corr, tot = 0, 0
    for xb, yb_mac, _yb_pos in loader:
        xb, yb_mac = xb.to(device), yb_mac.to(device)
        z = model.encode_csi(xb)                  # [B,d]
        pred = (z @ E.t()).argmax(-1)
        corr += (pred == yb_mac).sum().item()
        tot  += yb_mac.numel()
    return corr / max(1, tot)

def train_clip_pos(model, train_loader, val_loader, epochs=10, lr=1e-3, wd=1e-4,
               device="cuda", p_use_pos=0.5):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb_mac, yb_pos in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb_mac = yb_mac.to(device, non_blocking=True)
            yb_pos = yb_pos.to(device, non_blocking=True)

            use_pos = (torch.rand(()) < p_use_pos).item()
            logits = model(xb, yb_mac, yb_pos, use_pos=use_pos)
            loss = clip_loss(logits)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        sch.step()
        val_acc = eval_clip_mac_only(model, val_loader, device=device, mode="mac_only")
        print(f"[Epoch {ep:02d}] val@1(MAC-only)={val_acc*100:.2f}%  logit_scale={model.logit_scale.exp().item():.3f}")
        best_val = max(best_val, val_acc)

    print(f"Best Val@1 (MAC-only): {best_val*100:.2f}%")
    return model

# ================================================================
# Main
# ================================================================
def main():
    assert os.path.exists(TRAIN_CSV), f"Missing TRAIN_CSV: {TRAIN_CSV}"
    assert os.path.exists(TEST_CSV),  f"Missing TEST_CSV:  {TEST_CSV}"

    # 1) Load TRAIN with MAC+POS
    train_data, train_mac, train_pos = process_csv_fixed_id_uniform_sampling_rssi_pos(
        file_path=TRAIN_CSV,
        mac_id_list=MAC_ID_LIST,
        pos_tag_to_id=POS_TAG_TO_ID,
        max_samples_per_mac=100_000_000,
        stratify_by_pos=False,
        seed=SEED,
    )
    assert train_data is not None, "No valid TRAIN data loaded."

    # 2) MAC-stratified TRAIN/VAL split (REUSED splitter)
    tr_idx, va_idx = stratified_train_val_indices(train_mac, train_ratio=0.9, seed=SEED)
    print(f"[TRAIN CSV] TRAIN={len(tr_idx)} | VAL={len(va_idx)} (seed={SEED})")

    # 3) Build CLIP datasets/loaders (3-tuple with POS)
    clip_full_tr = CLIPDataset3(train_data, train_mac, train_pos, normalize=True, augment=True);  clip_full_tr.train_mode = True
    clip_full_va = CLIPDataset3(train_data, train_mac, train_pos, normalize=True, augment=False); clip_full_va.train_mode = False
    clip_train_loader, clip_val_loader = make_loaders_for_dataset(
        clip_full_tr, tr_idx, va_idx, batch_size=BATCH_TRAIN, num_workers=0
    )

    # 4) Load TEST with MAC+POS (shared evaluation set)
    random.seed(SEED)  # determinism inside loader sampling
    test_data, test_mac, test_pos = process_csv_fixed_id_uniform_sampling_rssi_pos(
        file_path=TEST_CSV,
        mac_id_list=MAC_ID_LIST,
        pos_tag_to_id=POS_TAG_TO_ID,
        max_samples_per_mac=1_000_000,
        stratify_by_pos=False,
        seed=SEED,
    )
    assert test_data is not None, "No valid TEST data loaded."
    clip_test_ds = CLIPDataset3(test_data, test_mac, test_pos, normalize=True, augment=False); clip_test_ds.train_mode=False
    clip_test_loader = DataLoader(clip_test_ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0, pin_memory=True)
    print(f"[TEST CSV] TEST={len(test_mac)}")

    # 5) Build model (ResNet-50 CSI encoder + MAC+POS label encoder)
    num_classes = len(MAC_ID_LIST)
    num_pos = len(POS_TAG_TO_ID)
    model = CSI_CLIP(
        csi_encoder=CSICNNEncoder(proj_dim=256, in_ch=2),
        label_encoder=MacPosEncoder(num_classes=num_classes, num_pos=num_pos, dim=256, learnable_w=True, init_w=0.5),
    )

    # 6) Train
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_clip(
        model, clip_train_loader, clip_val_loader,
        epochs=EPOCHS, lr=LR, wd=WEIGHT_DECAY, device=device, p_use_pos=P_USE_POS
    )

    # 7) Evaluate on TEST (MAC-only)
    test_acc_mac_only = eval_clip_mac_only(model, clip_test_loader, device=device, mode="mac_only")
    print(f"[CLIP+POS] TEST top-1 (MAC-only): {test_acc_mac_only*100:.2f}%")

    # Optional: evaluate averaging over positions (prompts averaged)
    test_acc_avgpos = eval_clip_mac_only(model, clip_test_loader, device=device, mode="avg_over_pos")
    print(f"[CLIP+POS] TEST top-1 (avg over POS prompts): {test_acc_avgpos*100:.2f}%")

    # 8) Save checkpoint
    torch.save({"model": model.state_dict()}, "best_csi_clip_macpos.pth")
    print("Checkpoint saved: best_csi_clip_macpos.pth")

if __name__ == "__main__":
    main()


#%%

# ================================================================
# One set of reproducible splits (seeded) used by:
#   1) SimpleCNN (classifier)
#   2) ResNet50CSI (classifier)
#   3) CSI_CLIP (MAC-only)
#   4) CSI_CLIP (MAC + POS)
# ================================================================

import os, csv, math, random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
from torchvision import models
from torchvision.models import ResNet50_Weights

# --------------------
# USER SETTINGS
# --------------------
SEED = 20250910

# Your EXISTING combined CSVs (no headers), with POS column present
TRAIN_CSV = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\train\train_with_pos.csv"
TEST_CSV  = r"C:\Users\fawaz\OneDrive - University of South Florida\Desktop\csi_expts_from_jul_2025\openai_clip_based\individual_positions\p1_p2_p3_p4_p5_p6_p7_p8\test\test_with_pos.csv"

# MAC classes (order = label id)
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
POS_TAG_TO_ID = {f"p{i}": i-1 for i in range(1, 9)}  # p1->0 ... p8->7

# Train/eval config
EPOCHS = 10
BATCH_TRAIN = 64
BATCH_EVAL  = 64
LR_CLASSIFIER = 1e-3
LR_CLIP       = 1e-3
WD            = 1e-4
P_USE_POS     = 0.5  # prob. to use position in CLIP+POS training

RUN_SIMPLECNN     = True
RUN_RESNET50CSI   = False
RUN_CLIP_MAC_ONLY = True
RUN_CLIP_MAC_POS  = True

# --------------------
# Reproducibility
# --------------------
def set_global_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_global_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Loader -> (data [N,64,2], mac_labels [N], pos_labels [N])
# (Mirrors your rssi loader; **REUSED idea**, extended for POS)
# ================================================================
def parse_csi_data(csi_row: str):
    vals = csi_row.split()
    if len(vals) != 128:
        return None
    out = []
    for i in range(0, 128, 2):
        try:
            mag = float(vals[i]); ang = float(vals[i+1])
        except ValueError:
            return None
        out.append([mag, ang])
    return torch.tensor(out)  # [64,2]

def process_csv_fixed_id_uniform_sampling_rssi_pos(
    file_path: str,
    mac_id_list: list,
    pos_tag_to_id: dict = None,
    max_samples_per_mac: int = 50_000,
    stratify_by_pos: bool = False,
    seed: int = SEED,
):
    """
    Reads CSV with 3 or 5 columns:
      - 3: [MAC, CSI, POS]
      - 5: [MAC, RSSI, NOISE, CSI, POS]
    Returns:
      data:       [N, 64, 2] float32
      mac_labels: [N] long (0..C-1)
      pos_labels: [N] long (0..K-1)
    """
    random.seed(seed)
    mac_to_label = {m: i for i, m in enumerate(mac_id_list)}
    if pos_tag_to_id is None:
        pos_tag_to_id = {f"p{i}": i-1 for i in range(1, 9)}

    store = defaultdict(list)  # key: mac_id or (mac_id,pos_id)
    with open(file_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) not in (3, 5):
                continue
            if len(row) == 3:
                mac, csi_row, pos = row
            else:
                mac, _, _, csi_row, pos = row

            if mac not in mac_to_label:
                continue

            pos_lc = str(pos).strip().lower()
            if pos_lc not in pos_tag_to_id:
                pl = pos_lc.lstrip("p")
                if pl.isdigit():
                    pos_lc = f"p{int(pl)}"
                if pos_lc not in pos_tag_to_id:
                    continue

            csi_t = parse_csi_data(csi_row)
            if csi_t is None:
                continue

            mac_id = mac_to_label[mac]
            pos_id = pos_tag_to_id[pos_lc]
            key = (mac_id, pos_id) if stratify_by_pos else mac_id
            store[key].append((csi_t, mac_id, pos_id))

    data, mac_labels, pos_labels = [], [], []
    if stratify_by_pos:
        by_mac_total = defaultdict(int)
        for (mac_id, pos_id), entries in store.items():
            budget = max_samples_per_mac - by_mac_total[mac_id]
            if budget <= 0: continue
            take = min(len(entries), budget)
            for (csi_t, m, p) in random.sample(entries, take):
                data.append(csi_t); mac_labels.append(m); pos_labels.append(p)
                by_mac_total[mac_id] += 1
    else:
        for mac_id, entries in store.items():
            take = min(len(entries), max_samples_per_mac)
            for (csi_t, m, p) in random.sample(entries, take):
                data.append(csi_t); mac_labels.append(m); pos_labels.append(p)

    if not data:
        return None, None, None

    data = torch.stack(data).squeeze().float()               # [N,64,2]
    mac_labels = torch.tensor(mac_labels, dtype=torch.long)  # [N]
    pos_labels = torch.tensor(pos_labels, dtype=torch.long)  # [N]
    return data, mac_labels, pos_labels


# ================================================================
# Shared split utils (**REUSED** from earlier)
# ================================================================
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
        n_train = min(max(n_train, 1 if n > 1 else n), n-1 if n > 1 else n)
        train_idx += idxs[:n_train]
        val_idx   += idxs[n_train:]
    train_idx.sort(); val_idx.sort()
    return train_idx, val_idx

def make_loaders_for_dataset(dataset, tr_idx, va_idx, batch_size=64, num_workers=0, drop_last_train=True):
    ds_train = Subset(dataset, tr_idx)
    ds_val   = Subset(dataset, va_idx)
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True, drop_last=drop_last_train)
    val_loader   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ================================================================
# Datasets (same tensors, different views)
# ================================================================
class CLIPDataset2(Dataset):
    """(x, mac) — MAC-only CLIP.  Normalization/aug as before (**REUSED pattern**)."""
    def __init__(self, x_64x2, y_mac, normalize=True, augment=False):
        self.x = x_64x2.permute(0,2,1).float()  # [N,2,64]
        self.y_mac = y_mac.long()
        self.normalize, self.augment = normalize, augment
        self._train = False
    def __len__(self): return self.x.size(0)
    def _norm(self, x):
        m = x.mean(dim=-1, keepdim=True); s = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - m) / s
    def _aug(self, x):
        return x + 0.01*torch.randn_like(x) if torch.rand(()) < 0.5 else x
    @property
    def train_mode(self): return self._train
    @train_mode.setter
    def train_mode(self, v): self._train = bool(v)
    def __getitem__(self, i):
        xi = self._norm(self.x[i]) if self.normalize else self.x[i]
        if self.augment and self._train: xi = self._aug(xi)
        return xi, self.y_mac[i]

class CLIPDataset3(Dataset):
    """(x, mac, pos) — MAC+POS CLIP.  (**REUSED pattern**)."""
    def __init__(self, x_64x2, y_mac, y_pos, normalize=True, augment=False):
        self.x = x_64x2.permute(0,2,1).float()
        self.y_mac = y_mac.long()
        self.y_pos = y_pos.long()
        self.normalize, self.augment = normalize, augment
        self._train = False
    def __len__(self): return self.x.size(0)
    def _norm(self, x):
        m = x.mean(dim=-1, keepdim=True); s = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - m) / s
    def _aug(self, x):
        return x + 0.01*torch.randn_like(x) if torch.rand(()) < 0.5 else x
    @property
    def train_mode(self): return self._train
    @train_mode.setter
    def train_mode(self, v): self._train = bool(v)
    def __getitem__(self, i):
        xi = self._norm(self.x[i]) if self.normalize else self.x[i]
        if self.augment and self._train: xi = self._aug(xi)
        return xi, self.y_mac[i], self.y_pos[i]

class CNNClsDataset(Dataset):
    """For SimpleCNN/ResNet50CSI classifiers — returns (x[1,64,2], mac)."""
    def __init__(self, x_64x2, y_mac, normalize=True):
        x = x_64x2.unsqueeze(1).float()  # [N,1,64,2]
        if normalize:
            m = x.mean(dim=(2,3), keepdim=True)
            s = x.std (dim=(2,3), keepdim=True).clamp_min(1e-6)
            x = (x - m) / s
        self.x = x
        self.y = y_mac.long()
    def __len__(self): return self.x.size(0)
    def __getitem__(self, i): return self.x[i], self.y[i]


# ================================================================
# Models
# ================================================================
# --- (A) Classifiers ---
class SimpleCNN(nn.Module):
    """Tiny 2D CNN for [B,1,64,2] -> logits."""
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,32,1,1]
        )
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        h = self.net(x).flatten(1)   # [B,32]
        return self.fc(h)

class ResNet50CSI(nn.Module):
    """
    Your resize-to-224 adapter (unchanged), for classification.
    INPUT:  x [B,1,64,2]  -> upsample to 224x224, repeat to 3ch
    OUTPUT: logits [B,num_classes]
    """
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)  # [B,1,224,224]
        x = x.repeat(1, 3, 1, 1)  # [B,3,224,224]
        return self.backbone(x)

# --- (B) CLIP encoders/label side ---
class CSICNNEncoder(nn.Module):
    """1D CNN encoder for CLIP — **REUSED** from earlier CNN CLIP."""
    def __init__(self, proj_dim=256, in_ch=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
    def forward(self, x_2x64: torch.Tensor) -> torch.Tensor:
        h = self.net(x_2x64).squeeze(-1)   # [B,128]
        z = self.ln(self.proj(h))          # [B,d]
        return F.normalize(z, dim=-1)

class LabelEmbedder(nn.Module):
    """MAC-only label embedder (**REUSED**)."""
    def __init__(self, num_classes, dim=256):
        super().__init__()
        self.emb = nn.Embedding(num_classes, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, y): return F.normalize(self.emb(y), dim=-1)
    def table(self):    return F.normalize(self.emb.weight, dim=-1)

class PositionEmbedder(nn.Module):
    def __init__(self, num_pos, dim=256):
        super().__init__()
        self.emb = nn.Embedding(num_pos, dim)
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, p): return F.normalize(self.emb(p), dim=-1)
    def table(self):     return F.normalize(self.emb.weight, dim=-1)

class MacPosEncoder(nn.Module):
    """z_text = normalize( z_mac + w * z_pos ) — from earlier CLIP+POS."""
    def __init__(self, num_classes, num_pos, dim=256, learnable_w=True, init_w=0.5):
        super().__init__()
        self.mac = LabelEmbedder(num_classes, dim)
        self.pos = PositionEmbedder(num_pos, dim)
        if learnable_w:
            self._logit_w = nn.Parameter(torch.tensor(math.log(init_w/(1-init_w))))
        else:
            self.register_buffer("_w", torch.tensor(float(init_w)))
    def weight(self):
        return torch.sigmoid(self._logit_w) if hasattr(self, "_logit_w") else self._w
    def forward(self, mac_y, pos_y=None, use_pos=True):
        z = self.mac(mac_y)
        if use_pos and (pos_y is not None):
            z = F.normalize(z + self.weight() * self.pos(pos_y), dim=-1)
        return z
    # Class tables
    def class_table_mac_only(self):  return self.mac.table()
    def class_table_avg_over_pos(self):
        Emac = self.mac.table(); Epos = self.pos.table(); w = self.weight()
        M = F.normalize(Emac[:, None, :] + w * Epos[None, :, :], dim=-1)  # [C,K,d]
        return F.normalize(M.mean(dim=1), dim=-1)

class CSI_CLIP(nn.Module):
    def __init__(self, csi_encoder: nn.Module, label_encoder: nn.Module):
        super().__init__()
        self.csi = csi_encoder
        self.txt = label_encoder
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def forward(self, csi_batch, mac_y, pos_y=None, use_pos=True):
        zc = self.csi(csi_batch)  # [B, d]

        # If the label encoder supports position (MacPosEncoder), pass pos.
        # Otherwise (LabelEmbedder), call with MAC only.
        if hasattr(self.txt, "class_table_avg_over_pos"):
            zt = self.txt(mac_y, pos_y, use_pos=use_pos)     # Mac+Pos
        else:
            zt = self.txt(mac_y)                             # MAC-only

        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * zc @ zt.t()  # [B, B]

    @torch.no_grad()
    def encode_csi(self, csi_batch):
        return self.csi(csi_batch)

    @torch.no_grad()
    def class_table(self, mode="mac_only"):
        # MAC-only always works
        if mode == "mac_only":
            if hasattr(self.txt, "class_table_mac_only"):
                return self.txt.class_table_mac_only()
            else:
                return self.txt.table()  # LabelEmbedder fallback

        # avg_over_pos only if the label encoder supports it
        if mode == "avg_over_pos" and hasattr(self.txt, "class_table_avg_over_pos"):
            return self.txt.class_table_avg_over_pos()

        # Fallback to MAC-only if requested mode isn't supported
        if hasattr(self.txt, "class_table_mac_only"):
            return self.txt.class_table_mac_only()
        return self.txt.table()


# ================================================================
# Training/Eval loops
# ================================================================
# --- Classification (SimpleCNN / ResNet50CSI) ---
def train_classifier(model, train_loader, val_loader, epochs=10, lr=1e-3, wd=1e-4, device=device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        acc = eval_classifier(model, val_loader, device=device)
        best = max(best, acc)
        print(f"[Epoch {ep:02d}] val@1={acc*100:.2f}%")
    return model, best

@torch.no_grad()
def eval_classifier(model, loader, device=device):
    model.eval()
    corr = tot = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=-1)
        corr += (pred == yb).sum().item(); tot += yb.numel()
    return corr / max(1, tot)

# --- CLIP (REUSED loss/eval structure) ---
def clip_loss(logits):
    B = logits.size(0)
    tgt = torch.arange(B, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt))

def train_clip_mac_only(model, train_loader, val_loader, epochs=10, lr=1e-3, wd=1e-4, device=device):
    """For MAC-only CLIP: loader yields (x, mac)."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb_mac in train_loader:
            xb, yb_mac = xb.to(device), yb_mac.to(device)
            logits = model(xb, yb_mac, None, use_pos=False)
            loss = clip_loss(logits)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        acc = eval_clip_mac_only(model, val_loader, device=device, mode="mac_only")
        best = max(best, acc)
        print(f"[Epoch {ep:02d}] val@1(MAC-only)={acc*100:.2f}%  logit_scale={model.logit_scale.exp().item():.3f}")
    return model, best

def train_clip_mac_pos(model, train_loader, val_loader, epochs=10, lr=1e-3, wd=1e-4, p_use_pos=0.5, device=device):
    """For MAC+POS CLIP: loader yields (x, mac, pos) and we randomly drop POS."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = 0.0
    for ep in range(1, epochs+1):
        model.train()
        for xb, yb_mac, yb_pos in train_loader:
            xb, yb_mac, yb_pos = xb.to(device), yb_mac.to(device), yb_pos.to(device)
            use_pos = (torch.rand(()) < p_use_pos).item()
            logits = model(xb, yb_mac, yb_pos, use_pos=use_pos)
            loss = clip_loss(logits)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
        acc = eval_clip_mac_only(model, val_loader, device=device, mode="mac_only")
        best = max(best, acc)
        print(f"[Epoch {ep:02d}] val@1(MAC-only)={acc*100:.2f}%  logit_scale={model.logit_scale.exp().item():.3f}")
    return model, best

@torch.no_grad()
def eval_clip_mac_only(model, loader, device=device, mode="mac_only"):
    model.eval()
    E = model.class_table(mode=mode).to(device)   # [C,d]
    corr = tot = 0
    for batch in loader:
        if len(batch) == 3:
            xb, yb_mac, _ = batch
        else:
            xb, yb_mac = batch
        xb, yb_mac = xb.to(device), yb_mac.to(device)
        z = model.encode_csi(xb)
        pred = (z @ E.t()).argmax(-1)
        corr += (pred == yb_mac).sum().item(); tot += yb_mac.numel()
    return corr / max(1, tot)


# ================================================================
# Main — one shared split; four trainings using the same samples
# ================================================================
def main():
    assert os.path.exists(TRAIN_CSV), f"Missing TRAIN_CSV: {TRAIN_CSV}"
    assert os.path.exists(TEST_CSV),  f"Missing TEST_CSV:  {TEST_CSV}"

    # ---- Load TRAIN (MAC + POS) once ----
    train_x, train_mac, train_pos = process_csv_fixed_id_uniform_sampling_rssi_pos(
        file_path=TRAIN_CSV,
        mac_id_list=MAC_ID_LIST,
        pos_tag_to_id=POS_TAG_TO_ID,
        max_samples_per_mac=100_000_000,
        stratify_by_pos=False,
        seed=SEED,
    )
    assert train_x is not None, "No valid TRAIN data."

    # ---- Shared MAC-stratified TRAIN/VAL indices (REUSED splitter) ----
    tr_idx, va_idx = stratified_train_val_indices(train_mac, train_ratio=0.9, seed=SEED)
    print(f"[TRAIN CSV] TRAIN={len(tr_idx)} | VAL={len(va_idx)} (seed={SEED})")

    # ---- Build datasets/loaders for each pipeline using the SAME indices ----
    # (a) Classifiers: SimpleCNN/ResNet50CSI
    ds_cls_full = CNNClsDataset(train_x, train_mac, normalize=True)
    cls_train_loader, cls_val_loader = make_loaders_for_dataset(
        ds_cls_full, tr_idx, va_idx, batch_size=BATCH_TRAIN, num_workers=0, drop_last_train=True
    )

    # (b) CLIP MAC-only
    ds_clip2_tr = CLIPDataset2(train_x, train_mac, normalize=True, augment=True);  ds_clip2_tr.train_mode = True
    ds_clip2_va = CLIPDataset2(train_x, train_mac, normalize=True, augment=False); ds_clip2_va.train_mode = False
    clip2_train_loader, clip2_val_loader = make_loaders_for_dataset(
        ds_clip2_tr, tr_idx, va_idx, batch_size=BATCH_TRAIN, num_workers=0, drop_last_train=True
    )

    # (c) CLIP MAC+POS
    ds_clip3_tr = CLIPDataset3(train_x, train_mac, train_pos, normalize=True, augment=True);  ds_clip3_tr.train_mode = True
    ds_clip3_va = CLIPDataset3(train_x, train_mac, train_pos, normalize=True, augment=False); ds_clip3_va.train_mode = False
    clip3_train_loader, clip3_val_loader = make_loaders_for_dataset(
        ds_clip3_tr, tr_idx, va_idx, batch_size=BATCH_TRAIN, num_workers=0, drop_last_train=True
    )

    # ---- Load TEST once; reuse for all pipelines ----
    random.seed(SEED)  # determinism inside loader sampling
    test_x, test_mac, test_pos = process_csv_fixed_id_uniform_sampling_rssi_pos(
        file_path=TEST_CSV,
        mac_id_list=MAC_ID_LIST,
        pos_tag_to_id=POS_TAG_TO_ID,
        max_samples_per_mac=1_000_000,
        stratify_by_pos=False,
        seed=SEED,
    )
    assert test_x is not None, "No valid TEST data."

    # Classifier TEST loader
    ds_cls_test = CNNClsDataset(test_x, test_mac, normalize=True)
    cls_test_loader = DataLoader(ds_cls_test, batch_size=BATCH_EVAL, shuffle=False, num_workers=0, pin_memory=True)

    # CLIP TEST loaders
    ds_clip2_test = CLIPDataset2(test_x, test_mac, normalize=True, augment=False); ds_clip2_test.train_mode = False
    clip2_test_loader = DataLoader(ds_clip2_test, batch_size=BATCH_EVAL, shuffle=False, num_workers=0, pin_memory=True)

    ds_clip3_test = CLIPDataset3(test_x, test_mac, test_pos, normalize=True, augment=False); ds_clip3_test.train_mode = False
    clip3_test_loader = DataLoader(ds_clip3_test, batch_size=BATCH_EVAL, shuffle=False, num_workers=0, pin_memory=True)

    num_classes = len(MAC_ID_LIST)
    num_pos     = len(POS_TAG_TO_ID)

    # ============================
    # 1) SimpleCNN (classifier)
    # ============================
    if RUN_SIMPLECNN:
        print("\n==> Training SimpleCNN (classifier) on shared split")
        simplecnn = SimpleCNN(num_classes).to(device)
        simplecnn, best_val = train_classifier(
            simplecnn, cls_train_loader, cls_val_loader,
            epochs=EPOCHS, lr=LR_CLASSIFIER, wd=WD, device=device
        )
        test_acc = eval_classifier(simplecnn, cls_test_loader, device=device)
        print(f"[SimpleCNN] TEST top-1: {test_acc*100:.2f}%")
        torch.save({"model": simplecnn.state_dict()}, "simplecnn_best.pth")

    # ============================
    # 2) ResNet50CSI (classifier)
    # ============================
    if RUN_RESNET50CSI:
        print("\n==> Training ResNet50CSI (classifier) on shared split")
        res50 = ResNet50CSI(num_classes=num_classes, pretrained=True).to(device)
        res50, best_val = train_classifier(
            res50, cls_train_loader, cls_val_loader,
            epochs=EPOCHS, lr=LR_CLASSIFIER, wd=WD, device=device
        )
        test_acc = eval_classifier(res50, cls_test_loader, device=device)
        print(f"[ResNet50CSI] TEST top-1: {test_acc*100:.2f}%")
        torch.save({"model": res50.state_dict()}, "resnet50csi_best.pth")

    # ============================
    # 3) CSI_CLIP (MAC-only)
    # ============================
    if RUN_CLIP_MAC_ONLY:
        print("\n==> Training CSI_CLIP (MAC-only) on shared split")
        clip_mac = CSI_CLIP(
            csi_encoder=CSICNNEncoder(proj_dim=256, in_ch=2),
            label_encoder=LabelEmbedder(num_classes=num_classes, dim=256),   # **REUSED MAC-only label side**
        )
        clip_mac, best_val = train_clip_mac_only(
            clip_mac, clip2_train_loader, clip2_val_loader,
            epochs=EPOCHS, lr=LR_CLIP, wd=WD, device=device
        )
        test_acc = eval_clip_mac_only(clip_mac, clip2_test_loader, device=device, mode="mac_only")
        print(f"[CSI_CLIP | MAC-only] TEST top-1: {test_acc*100:.2f}%")
        torch.save({"model": clip_mac.state_dict()}, "clip_mac_only_best.pth")

    # ============================
    # 4) CSI_CLIP (MAC + POS)
    # ============================
    if RUN_CLIP_MAC_POS:
        print("\n==> Training CSI_CLIP (MAC + POS) on shared split")
        clip_pos = CSI_CLIP(
            csi_encoder=CSICNNEncoder(proj_dim=256, in_ch=2),
            label_encoder=MacPosEncoder(num_classes=num_classes, num_pos=num_pos, dim=256, learnable_w=True, init_w=0.5),
        )
        clip_pos, best_val = train_clip_mac_pos(
            clip_pos, clip3_train_loader, clip3_val_loader,
            epochs=EPOCHS, lr=LR_CLIP, wd=WD, p_use_pos=P_USE_POS, device=device
        )
        test_acc_mac = eval_clip_mac_only(clip_pos, clip3_test_loader, device=device, mode="mac_only")
        print(f"[CSI_CLIP | MAC+POS] TEST top-1 (MAC-only prompts): {test_acc_mac*100:.2f}%")
        test_acc_avg = eval_clip_mac_only(clip_pos, clip3_test_loader, device=device, mode="avg_over_pos")
        print(f"[CSI_CLIP | MAC+POS] TEST top-1 (avg over POS prompts): {test_acc_avg*100:.2f}%")
        torch.save({"model": clip_pos.state_dict()}, "clip_mac_pos_best.pth")

    print("\nAll done. All four runs used the **same** train/val/test samples (seeded).")

if __name__ == "__main__":
    main()
   
    
    
    