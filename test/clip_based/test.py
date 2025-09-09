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
            nn.Conv1d(in_channels = in_ch, out_channels, = 64, kernel_size = 5, padding = 2), nn.ReLU(),
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

class LabelEncoder(nn.Module):
     """
    Learnable embedding per MAC/class. Works great when the class set is fixed.
    """
    
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
        scale - self.logit_scale.exp().clamp(max=100.0)
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
    
class CDIDataset(Dataset):
    
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
        std = x.std(dim = -1, keepdim = True).clamp_mim(1e-6)
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
            xi = self.augment(xi)
            
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



def train(model, train-ds, val_ds, * , epochs = 10, batch_size = 512, lr = 1e-3, wd = 1e4, num_workers = 0, device = 'cuda'):
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
    
    best-val = 0.0
    
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking = True)
            yb = yb.to(device, non_blocking = True)
            
            logits = model(xb, yb)
            loss = clip_loss(logits)
            
            opt.zero_grad(set_to_none = True)
            losss.backward()
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
    # file_path = input("Please enter the file path to the CSV file: ")
    # data, labels = process_csv_fixed_id_uniform_sampling_rssi(
    #     file_path=file_path,
    #     mac_id_list=[
    #         "00:FC:BA:38:4B:00",
    #         "70:0F:6A:BF:C1:40",
    #         "00:FC:BA:38:4B:01",
    #         "00:FC:BA:38:4B:02",
    #         "70:0F:6A:BF:C1:42",
    #         "FE:19:28:38:54:40",
    #         "70:0F:6A:FC:51:80",
    #         "70:0F:6A:FC:51:81",
    #         "70:0F:6A:E9:9D:81",
    #         "00:FC:BA:27:63:01",
    #         "70:0F:6A:FC:51:82",
    #         "00:FC:BA:27:63:61",
    #     ],
    #     max_samples_per_mac=40000
    # )
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
    trained = train(
        model, train_ds, val_ds,
        epochs=10, batch_size=512, lr=1e-3, wd=1e-4, num_workers=2, device="cuda"
    )

    # Final eval on the validation split (zero-shot style classification)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=2)
    val_acc = evaluate_zero_shot(trained, val_loader, device="cuda")
    print(f"Validation top-1 accuracy: {val_acc*100:.2f}%")
























    
    
    
    
    
    
    