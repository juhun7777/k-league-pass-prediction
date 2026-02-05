import pandas as pd
import numpy as np
import os
import math
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. Configuration
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

# Hybrid Loss Weight
L2_LAMBDA = 10.0

torch.manual_seed(SEED)
np.random.seed(SEED)

# Transformer Params
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FEEDFORWARD = 256
DROPOUT = 0.15
WINDOW_SIZE = 5  # [수정] 문맥 파악 강화 (3 -> 5)

# =============================================================================
# 2. Maps Update (핵심 수정 사항)
# =============================================================================
# [수정] 데이터에 존재하는 모든 이벤트 타입 반영 (기존 V20은 15개뿐이라 롱볼을 못 배움)
TYPE_MAP = [
    'Pass', 'Carry', 'Ball Recovery', 'Duel', 'Clearance', 'Block', 
    'Interception', 'Foul', 'Goal', 'Shot', 'Offside', 'Tackle', 
    'Substitution', 'Keeper Rush-Out', 'Aerial Clearance', 
    # --- 추가된 롱볼/특수 이벤트 ---
    'Cross', 'Goal Kick', 'Pass_Freekick', 'Pass_Corner', 'Throw-In',
    'Penalty Kick', 'Free Kick', 'Corner', 'Catch', 'Punch', 'Save',
    'Claim', 'Turnover', 'Take-On', 'Shield', 'Error', 'Intervention',
    'Deflection', 'Own Goal'
]

# [수정] 실제 데이터의 결과값으로 변경 (Success -> Successful)
RESULT_MAP = ['Successful', 'Unsuccessful']

# (참고: Team ID Map은 데이터 로딩 후 동적으로 생성)

def get_one_hot(value, mapping):
    vec = [0.0] * (len(mapping) + 1)
    try:
        # 데이터가 nan인 경우 (Carry 등 결과가 없는 경우)
        if pd.isna(value):
            vec[-1] = 1.0
            return vec
            
        if isinstance(mapping, list):
            if value in mapping:
                idx = mapping.index(value)
                vec[idx] = 1.0
            else:
                vec[-1] = 1.0
        elif isinstance(mapping, dict): # For Team ID
             idx = mapping.get(value, -1)
             if idx != -1: vec[idx] = 1.0
             else: vec[-1] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec

# =============================================================================
# 3. Feature Extraction
# =============================================================================
def extract_features(g):
    # g: DataFrame of a single episode
    if len(g) > WINDOW_SIZE:
        g = g.iloc[-WINDOW_SIZE:].reset_index(drop=True)

    sx = g["start_x"].values / 105.0
    sy = g["start_y"].values / 68.0
    ex = g["end_x"].values / 105.0
    ey = g["end_y"].values / 68.0
    times = g["time_seconds"].values
    is_home = g["is_home"].values.astype(float)
    type_names = g["type_name"].values
    result_names = g["result_name"].values
    
    # Team ID (선택사항: V20 원본에는 없었으나 성능 도움됨. 여기선 V20 구조 유지를 위해 제외하거나 포함 가능)
    # V20 원본 유지를 원하셨으므로 일단 제외하되, 맵핑 수정 효과만 봅니다.
    # (만약 Team ID 넣고 싶으시면 이 부분 주석 해제)
    # team_ids = g["team_id"].values 

    coords = []
    for i in range(len(g)):
        start_features = []
        
        # 1. Location
        start_features.extend([sx[i], sy[i]]) 

        # 2. Displacement (직전 이벤트 대비 이동이 아님, 해당 이벤트의 이동 벡터)
        # Carry나 Pass는 end_x가 있지만, 어떤 이벤트는 없을 수 있음 (전처리된 데이터 가정)
        if i < len(g) - 1:
            dx = ex[i] - sx[i]
            dy = ey[i] - sy[i]
        else:
            dx, dy = 0.0, 0.0 # 마지막 예측 대상은 dx, dy를 모름 (Target이니까)
        
        start_features.extend([dx, dy])

        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        start_features.extend([dist, angle])

        # 3. Time Delta
        time_delta = times[i] - times[i-1] if i > 0 else 0.0
        start_features.append(time_delta)

        if time_delta > 0:
            vx = dx / time_delta
            vy = dy / time_delta
            speed = dist / time_delta
        else:
            vx, vy, speed = 0.0, 0.0, 0.0
        start_features.extend([vx, vy, speed])

        # 4. Meta
        start_features.append(is_home[i])
        
        # 5. Type & Result (수정된 맵핑 적용)
        start_features.extend(get_one_hot(type_names[i], TYPE_MAP))
        start_features.extend(get_one_hot(result_names[i], RESULT_MAP))
        
        # 6. Goal Context
        curr_x, curr_y = sx[i] * 105.0, sy[i] * 68.0
        goal_dist = np.sqrt((105.0 - curr_x)**2 + (34.0 - curr_y)**2) / 105.0
        goal_angle = np.arctan2(34.0 - curr_y, 105.0 - curr_x)
        center_dist = np.sqrt((52.5 - curr_x)**2 + (34.0 - curr_y)**2) / 105.0
        
        start_features.extend([goal_dist, goal_angle, center_dist])
        
        coords.append(start_features)

    seq = np.array(coords, dtype="float32")
    target = np.array([ex[-1], ey[-1]], dtype="float32")
    return seq, target

# =============================================================================
# 4. Data Loading
# =============================================================================
print("Loading Data...")
df = pd.read_csv(TRAIN_FILE)
df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)

# [중요] 훈련 데이터에 없는 타입이 테스트에 나올 수 있으므로, TYPE_MAP은 고정해서 씀.
# 대신 데이터에 있는 것 중 TYPE_MAP에 없는 게 있는지 확인 (디버깅용)
uniq_types = df['type_name'].unique()
missing_types = [t for t in uniq_types if t not in TYPE_MAP and isinstance(t, str)]
if missing_types:
    print(f"Warning: Still missing types in Map: {missing_types}")
    # 발견되면 Map에 자동 추가 (안전장치)
    TYPE_MAP.extend(missing_types)

episodes, targets, episode_ids = [], [], []

for epi_id, g in tqdm(df.groupby("game_episode"), desc="Processing Episodes"):
    g = g.reset_index(drop=True)
    if len(g) < 2: continue
    seq, target = extract_features(g)
    episodes.append(seq)
    targets.append(target)
    episode_ids.append(epi_id)

print(f"Total episodes: {len(episodes)}")
INPUT_DIM = episodes[0].shape[1]
print(f"Input Feature Dimension: {INPUT_DIM}") # 맵 확장으로 차원 증가 예상

episodes = np.array(episodes, dtype=object)
targets = np.array(targets, dtype=np.float32)
episode_ids = np.array(episode_ids)

def augment_data(seqs, tgts):
    aug_seqs, aug_tgts = [], []
    for seq, tgt in zip(seqs, tgts):
        aug_seqs.append(seq)
        aug_tgts.append(tgt)
        
        # Flip Augmentation
        # Feature Index:
        # 0:sx, 1:sy(Flip), 2:dx, 3:dy(Flip), 4:dist, 5:angle(Flip)
        # 6:dt, 7:vx, 8:vy(Flip), 9:spd
        # 10:is_home
        # ... One-hots ...
        # End-3: goal_dist, End-2: goal_angle(Flip), End-1: center_dist
        
        seq_aug = seq.copy()
        tgt_aug = tgt.copy()
        
        seq_aug[:, 1] = 1.0 - seq_aug[:, 1] # sy
        seq_aug[:, 3] = -seq_aug[:, 3]      # dy
        seq_aug[:, 5] = -seq_aug[:, 5]      # angle
        seq_aug[:, 8] = -seq_aug[:, 8]      # vy
        
        # Goal Angle Flip
        if seq_aug.shape[1] > 20: 
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
        return torch.tensor(self.episodes[idx], dtype=torch.float32), \
               torch.tensor(self.targets[idx], dtype=torch.float32)

def collate_fn(batch):
    seqs, tgts = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    tgts = torch.stack(tgts, dim=0)
    return padded, lengths, tgts

# =============================================================================
# 5. Bivariate Loss (V20 Original)
# =============================================================================
class BivariateGaussianNLLLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_mu, pred_sigma, pred_rho, target):
        x_true = target[:, 0]
        y_true = target[:, 1]
        mu_x = pred_mu[:, 0]
        mu_y = pred_mu[:, 1]
        sigma_x = pred_sigma[:, 0]
        sigma_y = pred_sigma[:, 1]
        rho = pred_rho.squeeze(-1)
        
        z_norm_x = (x_true - mu_x) / sigma_x
        z_norm_y = (y_true - mu_y) / sigma_y
        z = (z_norm_x**2 + z_norm_y**2 - 2 * rho * z_norm_x * z_norm_y)
        rho_sq_compl = 1 - rho**2 + self.eps
        log_term = torch.log(2 * math.pi * sigma_x * sigma_y * torch.sqrt(rho_sq_compl))
        nll = log_term + (z / (2 * rho_sq_compl))
        
        return torch.mean(nll)

# =============================================================================
# 6. Model (V20 Original Transformer)
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
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerBivariate(nn.Module):
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
            nn.Linear(d_model // 2, 5) 
        )
    def forward(self, x, lengths):
        batch_size, max_len, _ = x.size()
        batch_indices = torch.arange(batch_size, device=x.device)
        start_pos = x[batch_indices, lengths-1, 0:2]
        
        mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        x_emb = self.input_proj(x)
        x_emb = self.pos_encoder(x_emb)
        output = self.transformer_encoder(x_emb, src_key_padding_mask=mask)
        
        last_outputs = output[batch_indices, lengths-1, :]
        out = self.fc(last_outputs)
        
        mu_delta = out[:, 0:2]
        sigma = F.softplus(out[:, 2:4]) + 1e-6 
        rho = torch.tanh(out[:, 4]).unsqueeze(-1)
        
        mu_abs = start_pos + mu_delta
        return mu_abs, sigma, rho

# =============================================================================
# 7. Training Loop
# =============================================================================
gkf = GroupKFold(n_splits=N_FOLDS)
fold_best_models = []

print(f"Start Training (V23 Data Aligned)...")

bivariate_nll = BivariateGaussianNLLLoss().to(DEVICE)

for fold, (train_idx, val_idx) in enumerate(gkf.split(episodes, targets, groups=episode_ids)):
    print(f"\n[{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}]")
    
    episodes_train, targets_train = augment_data(episodes[train_idx], targets[train_idx])
    train_loader = DataLoader(EpisodeDataset(episodes_train, targets_train), 
                              batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(EpisodeDataset(episodes[val_idx], targets[val_idx]), 
                              batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TransformerBivariate(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_dist = float("inf")
    best_state = None
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_accum = 0.0
        
        for X, lengths, y in tqdm(train_loader, desc=f"Ep {epoch}", leave=False):
            X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            
            pred_mu, pred_sigma, pred_rho = model(X, lengths)
            
            loss_nll = bivariate_nll(pred_mu, pred_sigma, pred_rho, y)
            loss_dist = torch.mean(torch.sqrt(torch.sum((pred_mu - y)**2, dim=1)))
            loss = loss_nll + (L2_LAMBDA * loss_dist)
            
            loss.backward()
            optimizer.step()
            train_loss_accum += loss.item() * X.size(0)
            
        train_loss_accum /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        dists = []
        with torch.no_grad():
            for X, lengths, y in valid_loader:
                X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                pred_mu, _, _ = model(X, lengths)
                pred_real = pred_mu.cpu().numpy() * [105.0, 68.0]
                true_real = y.cpu().numpy() * [105.0, 68.0]
                dists.extend(np.sqrt(np.sum((pred_real - true_real)**2, axis=1)))
        
        mean_dist = np.mean(dists)
        scheduler.step(mean_dist)
        
        if mean_dist < best_dist:
            best_dist = mean_dist
            best_state = model.state_dict().copy()
        
        current_lr = optimizer.param_groups[0]['lr']
        if mean_dist == best_dist or epoch % 5 == 0:
             print(f" Ep {epoch}: Loss={train_loss_accum:.4f} | Val Dist={mean_dist:.4f} | LR={current_lr:.6f}")
             
    print(f" >> Fold {fold+1} Best Dist: {best_dist:.4f}")
    
    # ✅ 가중치 저장 (추가된 유일한 부분)
    save_path = os.path.join(WEIGHTS_DIR, f"v23_fold{fold+1}.pth")
    torch.save(best_state, save_path)
    print(f"Saved V23 Fold {fold+1} to {save_path}")
    
    final_model = TransformerBivariate(INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT).to(DEVICE)
    final_model.load_state_dict(best_state)
    fold_best_models.append(final_model)