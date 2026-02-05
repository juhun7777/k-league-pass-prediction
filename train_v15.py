"""
Transformer V15: Goal Context Features (Fixed & Simple Average)
---------------------------------------------------------------
1. Fix: 'submission' variable definition order corrected.
2. Change: Removed Weighted Ensemble -> Uses Simple Mean of 5 Folds.
"""

import pandas as pd
import numpy as np
import os
import math
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. Configuration & Hyperparameters
# =============================================================================

BASE_PATH = "."
TRAIN_FILE = os.path.join(BASE_PATH, "train.csv")
WEIGHTS_DIR = os.path.join(BASE_PATH, "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-3
SEED = 42
N_FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# Transformer Hyperparameters
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 128
DROPOUT = 0.123

WINDOW_SIZE = 3

# =============================================================================
# 2. Feature Definitions
# =============================================================================

TYPE_MAP = [
    'Pass', 'Carry', 'Interception', 'Clearance', 'Duel', 'Recovery',
    'Shot', 'Goal', 'Foul', 'Offside', 'Tackle', 'Substitution',
    'Keeper Rush-Out', 'Block', 'Aerial Clearance'
]

RESULT_MAP = ['Success', 'Fail', 'Offside', 'Own Goal', 'Unsuccessful', 'Yellow_Card', 'Red_Card']
#RESULT_MAP = ['Successful',  'Unsuccessful']

def get_one_hot(value, mapping):
    vec = [0.0] * (len(mapping) + 1)
    try:
        idx = mapping.index(value)
        vec[idx] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec

# =============================================================================
# 3. Feature Extraction (Enhanced with Goal Context)
# =============================================================================

def extract_features(g):
    if len(g) > WINDOW_SIZE:
        g = g.iloc[-WINDOW_SIZE:].reset_index(drop=True)

    # Raw coordinates for context calculation
    raw_sx = g["start_x"].values
    raw_sy = g["start_y"].values
    
    # Normalized coordinates
    sx = raw_sx / 105.0
    sy = raw_sy / 68.0
    ex = g["end_x"].values / 105.0
    ey = g["end_y"].values / 68.0
    
    times = g["time_seconds"].values
    is_home = g["is_home"].values.astype(float)
    type_names = g["type_name"].values
    result_names = g["result_name"].values
    
    coords = []
    
    for i in range(len(g)):
        start_features = []
        # 1. Basic Coords
        start_features.extend([sx[i], sy[i]]) 
        
        # 2. Delta
        if i < len(g) - 1:
            dx = ex[i] - sx[i]
            dy = ey[i] - sy[i]
        else:
            dx, dy = 0.0, 0.0
        start_features.extend([dx, dy])
        
        # 3. Dist/Angle (Movement)
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        start_features.extend([dist, angle])
        
        # 4. Time/Speed
        time_delta = times[i] - times[i-1] if i > 0 else 0.0
        start_features.append(time_delta)
        
        if time_delta > 0:
            vx = dx / time_delta
            vy = dy / time_delta
            speed = dist / time_delta
        else:
            vx, vy, speed = 0.0, 0.0, 0.0
        start_features.extend([vx, vy, speed])
        
        # 5. Type/Result/Home
        start_features.append(is_home[i])
        start_features.extend(get_one_hot(type_names[i], TYPE_MAP))
        start_features.extend(get_one_hot(result_names[i], RESULT_MAP))
        
        # --- [NEW] Goal Context Features ---
        curr_x, curr_y = raw_sx[i], raw_sy[i]
        
        # Goal Distance (Target: 105, 34)
        goal_dist = np.sqrt((105.0 - curr_x)**2 + (34.0 - curr_y)**2)
        start_features.append(goal_dist / 105.0) # Normalize
        
        # Goal Angle
        goal_angle = np.arctan2(34.0 - curr_y, 105.0 - curr_x)
        start_features.append(goal_angle) # Radians
        
        # Center Distance (Target: 52.5, 34)
        center_dist = np.sqrt((52.5 - curr_x)**2 + (34.0 - curr_y)**2)
        start_features.append(center_dist / 105.0)
        
        coords.append(start_features)
            
    seq = np.array(coords, dtype="float32")
    target = np.array([ex[-1], ey[-1]], dtype="float32")
    
    return seq, target

# =============================================================================
# 4. Data Loading & Augmentation
# =============================================================================

print("Loading Train Data...")
df = pd.read_csv(TRAIN_FILE)
df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)

episodes = []
targets = []
episode_ids = []

for epi_id, g in tqdm(df.groupby("game_episode"), desc="Processing Episodes"):
    g = g.reset_index(drop=True)
    if len(g) < 2: continue
    seq, target = extract_features(g)
    episodes.append(seq)
    targets.append(target)
    episode_ids.append(epi_id)

print(f"Total episodes: {len(episodes)}")
INPUT_DIM = episodes[0].shape[1]
print(f"Input Feature Dimension: {INPUT_DIM}")

episodes = np.array(episodes, dtype=object)
targets = np.array(targets, dtype=object)
episode_ids = np.array(episode_ids)

def augment_data(seqs, tgts):
    aug_seqs = []
    aug_tgts = []
    
    for seq, tgt in zip(seqs, tgts):
        aug_seqs.append(seq)
        aug_tgts.append(tgt)
        
        seq_aug = seq.copy()
        tgt_aug = tgt.copy()
        
        # Flip Features: Y-axis reflection
        seq_aug[:, 1] = 1.0 - seq_aug[:, 1]
        seq_aug[:, 3] = -seq_aug[:, 3]
        seq_aug[:, 5] = -seq_aug[:, 5]
        seq_aug[:, 8] = -seq_aug[:, 8]
        
        # [Context Feature Flip]
        # Goal Angle (Index -2): arctan2(dy, dx) -> arctan2(-dy, dx) = -angle
        seq_aug[:, -2] = -seq_aug[:, -2]
        
        tgt_aug[1] = 1.0 - tgt_aug[1]
        
        aug_seqs.append(seq_aug)
        aug_tgts.append(tgt_aug)
        
    return aug_seqs, aug_tgts

class EpisodeDataset(Dataset):
    def __init__(self, episodes, targets):
        self.episodes = episodes
        self.targets = targets
    def __len__(self): return len(self.episodes)
    def __getitem__(self, idx): 
        seq = np.array(self.episodes[idx], dtype=np.float32)
        tgt = np.array(self.targets[idx], dtype=np.float32)
        return torch.tensor(seq), torch.tensor(tgt)

def collate_fn(batch):
    seqs, tgts = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    tgts = torch.stack(tgts, dim=0)
    return padded, lengths, tgts

# =============================================================================
# 5. Transformer Model
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBaseline(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x, lengths):
        batch_size, max_len, _ = x.size()
        mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        last_outputs = output[torch.arange(batch_size, device=x.device), lengths-1, :]
        prediction = self.fc(last_outputs)
        return prediction

class EuclideanLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.sqrt(torch.sum((pred - target)**2, dim=1) + self.eps).mean()

# =============================================================================
# 6. Training Loop (Simple Average Ensemble)
# =============================================================================

gkf = GroupKFold(n_splits=N_FOLDS)
fold_best_models = []

print(f"Start {N_FOLDS}-Fold Training (V15 Goal Context - Simple Average)...")

for fold, (train_idx, val_idx) in enumerate(gkf.split(episodes, targets, groups=episode_ids)):
    print(f"\n[{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}]")
    
    episodes_train_orig = episodes[train_idx]
    targets_train_orig = targets[train_idx]
    episodes_valid = episodes[val_idx]
    targets_valid = targets[val_idx]
    
    # Apply Augmentation
    episodes_train, targets_train = augment_data(episodes_train_orig, targets_train_orig)
    
    train_loader = DataLoader(EpisodeDataset(episodes_train, targets_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(EpisodeDataset(episodes_valid, targets_valid), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    model = TransformerBaseline(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(DEVICE)
    
    criterion = EuclideanLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    best_dist = float("inf")
    best_model_state = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_accum = 0.0
        
        for X, lengths, y in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
            X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X, lengths)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * X.size(0)
            
        train_loss_avg = train_loss_accum / len(train_loader.dataset)
        
        # Validation
        model.eval()
        dists = []
        with torch.no_grad():
            for X, lengths, y in valid_loader:
                X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                pred = model(X, lengths)
                
                pred_real = pred.cpu().numpy() * [105.0, 68.0]
                true_real = y.cpu().numpy() * [105.0, 68.0]
                
                batch_dists = np.sqrt(np.sum((pred_real - true_real)**2, axis=1))
                dists.extend(batch_dists)
        
        mean_dist = np.mean(dists)
        scheduler.step(mean_dist)
        
        if mean_dist < best_dist:
            best_dist = mean_dist
            best_model_state = model.state_dict().copy()
            
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0 or mean_dist == best_dist:
             print(f" Ep {epoch}: Loss={train_loss_avg:.4f} | Val Dist={mean_dist:.4f} | LR={current_lr:.6f}")
             
    print(f" >> Fold {fold+1} Best Dist: {best_dist:.4f}")
    
    # ✅ 가중치 저장 (추가된 유일한 부분)
    save_path = os.path.join(WEIGHTS_DIR, f"v15_fold{fold+1}.pth")
    torch.save(best_model_state, save_path)
    print(f"Saved V15 Fold {fold+1} to {save_path}")
    
    # Save Best Model
    final_model = TransformerBaseline(input_dim=INPUT_DIM, d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT).to(DEVICE)
    final_model.load_state_dict(best_model_state)
    fold_best_models.append(final_model)