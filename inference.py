import pandas as pd
import numpy as np
import os
import math
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

# =============================================================================
# 1. Configuration & Paths
# =============================================================================
# [제출 규정] 상대 경로
BASE_PATH = "."
TRAIN_FILE = os.path.join(BASE_PATH, "train.csv") # Map 동기화를 위해 필요
TEST_FILE = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_FILE = os.path.join(BASE_PATH, "sample_submission.csv")
WEIGHTS_DIR = os.path.join(BASE_PATH, "weights")
OUTPUT_FILE = os.path.join(BASE_PATH, "submission.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Ensemble Weights
W_V23 = 0.6
W_V15 = 0.4

print(f"Using Device: {DEVICE}")

# =============================================================================
# 2. Shared Modules
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

# =============================================================================
# 3. Model V23 Specifics (Window 5, Bivariate)
# =============================================================================
V23_WINDOW = 5
V23_TYPE_MAP = [
    'Pass', 'Carry', 'Ball Recovery', 'Duel', 'Clearance', 'Block', 
    'Interception', 'Foul', 'Goal', 'Shot', 'Offside', 'Tackle', 
    'Substitution', 'Keeper Rush-Out', 'Aerial Clearance', 
    # --- 추가된 롱볼/특수 이벤트 ---
    'Cross', 'Goal Kick', 'Pass_Freekick', 'Pass_Corner', 'Throw-In',
    'Penalty Kick', 'Free Kick', 'Corner', 'Catch', 'Punch', 'Save',
    'Claim', 'Turnover', 'Take-On', 'Shield', 'Error', 'Intervention',
    'Deflection', 'Own Goal'
]
# [수정] 학습 코드와 동일하게 Map 동기화 (Train 데이터에 있는 추가 타입 반영)
if os.path.exists(TRAIN_FILE):
    print("Syncing V23 Map with Train Data...")
    train_df_temp = pd.read_csv(TRAIN_FILE)
    uniq_types = train_df_temp['type_name'].unique()
    missing = [t for t in uniq_types if t not in V23_TYPE_MAP and isinstance(t, str)]
    if missing:
        print(f" >> Added missing types to V23: {missing}")
        V23_TYPE_MAP.extend(missing)
    del train_df_temp

V23_RESULT_MAP = ['Successful', 'Unsuccessful']

def get_one_hot_v23(value, mapping):
    vec = [0.0] * (len(mapping) + 1)
    try:
        if pd.isna(value):
            vec[-1] = 1.0
            return vec
        if value in mapping:
            vec[mapping.index(value)] = 1.0
        else:
            vec[-1] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec

def extract_features_v23(g):
    if len(g) > V23_WINDOW:
        g = g.iloc[-V23_WINDOW:].reset_index(drop=True)
    
    sx, sy = g["start_x"].values/105.0, g["start_y"].values/68.0
    ex, ey = g["end_x"].values/105.0, g["end_y"].values/68.0
    times = g["time_seconds"].values
    is_home = g["is_home"].values.astype(float)
    type_names, result_names = g["type_name"].values, g["result_name"].values
    
    coords = []
    for i in range(len(g)):
        feat = []
        feat.extend([sx[i], sy[i]])
        if i < len(g) - 1: dx, dy = ex[i]-sx[i], ey[i]-sy[i]
        else: dx, dy = 0.0, 0.0
        feat.extend([dx, dy])
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        feat.extend([dist, angle])
        
        dt = times[i] - times[i-1] if i > 0 else 0.0
        feat.append(dt)
        if dt > 0: feat.extend([dx/dt, dy/dt, dist/dt])
        else: feat.extend([0.0, 0.0, 0.0])
        
        feat.append(is_home[i])
        feat.extend(get_one_hot_v23(type_names[i], V23_TYPE_MAP))
        feat.extend(get_one_hot_v23(result_names[i], V23_RESULT_MAP))
        
        curr_x, curr_y = sx[i]*105.0, sy[i]*68.0
        goal_dist = np.sqrt((105.0-curr_x)**2 + (34.0-curr_y)**2)/105.0
        goal_angle = np.arctan2(34.0-curr_y, 105.0-curr_x)
        center_dist = np.sqrt((52.5-curr_x)**2 + (34.0-curr_y)**2)/105.0
        feat.extend([goal_dist, goal_angle, center_dist])
        coords.append(feat)
    return np.array(coords, dtype="float32")

class TransformerV23(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        self.pos_encoder = PositionalEncoding(128)
        encoder_layers = nn.TransformerEncoderLayer(128, 4, 256, 0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 3)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 5)
        )
    def forward(self, x, lengths):
        batch_size, max_len, _ = x.size()
        mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        x_emb = self.input_proj(x)
        x_emb = self.pos_encoder(x_emb)
        output = self.transformer_encoder(x_emb, src_key_padding_mask=mask)
        last_outputs = output[torch.arange(batch_size, device=x.device), lengths-1, :]
        out = self.fc(last_outputs)
        mu_delta = out[:, 0:2]
        batch_indices = torch.arange(batch_size, device=x.device)
        start_pos = x[batch_indices, lengths-1, 0:2]
        mu_abs = start_pos + mu_delta
        return mu_abs

# =============================================================================
# 4. Model V15 Specifics (Window 3, Euclidean)
# =============================================================================
V15_WINDOW = 3
V15_TYPE_MAP = [
    'Pass', 'Carry', 'Interception', 'Clearance', 'Duel', 'Recovery',
    'Shot', 'Goal', 'Foul', 'Offside', 'Tackle', 'Substitution',
    'Keeper Rush-Out', 'Block', 'Aerial Clearance'
]
V15_RESULT_MAP = ['Success', 'Fail', 'Offside', 'Own Goal', 'Unsuccessful', 'Yellow_Card', 'Red_Card']

def get_one_hot_v15(value, mapping):
    vec = [0.0] * (len(mapping) + 1)
    try:
        idx = mapping.index(value)
        vec[idx] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec

def extract_features_v15(g):
    if len(g) > V15_WINDOW:
        g = g.iloc[-V15_WINDOW:].reset_index(drop=True)

    raw_sx = g["start_x"].values
    raw_sy = g["start_y"].values
    sx, sy = raw_sx / 105.0, raw_sy / 68.0
    ex, ey = g["end_x"].values / 105.0, g["end_y"].values / 68.0
    times = g["time_seconds"].values
    is_home = g["is_home"].values.astype(float)
    type_names = g["type_name"].values
    result_names = g["result_name"].values
    
    coords = []
    for i in range(len(g)):
        feat = []
        feat.extend([sx[i], sy[i]])
        if i < len(g) - 1: dx, dy = ex[i]-sx[i], ey[i]-sy[i]
        else: dx, dy = 0.0, 0.0
        feat.extend([dx, dy])
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        feat.extend([dist, angle])
        
        dt = times[i] - times[i-1] if i > 0 else 0.0
        feat.append(dt)
        if dt > 0: feat.extend([dx/dt, dy/dt, dist/dt])
        else: feat.extend([0.0, 0.0, 0.0])
            
        feat.append(is_home[i])
        feat.extend(get_one_hot_v15(type_names[i], V15_TYPE_MAP))
        feat.extend(get_one_hot_v15(result_names[i], V15_RESULT_MAP))
        
        curr_x, curr_y = raw_sx[i], raw_sy[i]
        goal_dist = np.sqrt((105.0 - curr_x)**2 + (34.0 - curr_y)**2)
        feat.append(goal_dist / 105.0)
        goal_angle = np.arctan2(34.0 - curr_y, 105.0 - curr_x)
        feat.append(goal_angle)
        center_dist = np.sqrt((52.5 - curr_x)**2 + (34.0 - curr_y)**2)
        feat.append(center_dist / 105.0)
        coords.append(feat)
    return np.array(coords, dtype="float32")

class TransformerV15(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        self.pos_encoder = PositionalEncoding(128)
        encoder_layers = nn.TransformerEncoderLayer(128, 4, 128, 0.123, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 3)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.123),
            nn.Linear(64, 2)
        )
    def forward(self, x, lengths):
        batch_size, max_len, _ = x.size()
        mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        last_outputs = output[torch.arange(batch_size, device=x.device), lengths-1, :]
        return self.fc(last_outputs)

# =============================================================================
# 5. Inference Logic
# =============================================================================
def load_models(input_dim_v23, input_dim_v15):
    models_v23 = []
    models_v15 = []
    
    # Load V23 (Model 1)
    for i in range(1, 6):
        path = os.path.join(WEIGHTS_DIR, f"v23_fold{i}.pth")
        if os.path.exists(path):
            try:
                m = TransformerV23(input_dim_v23).to(DEVICE)
                m.load_state_dict(torch.load(path, map_location=DEVICE))
                m.eval()
                models_v23.append(m)
            except Exception as e:
                print(f"Error loading V23 Fold {i}: {e}")
                
    # Load V15 (Model 2)
    for i in range(1, 6):
        path = os.path.join(WEIGHTS_DIR, f"v15_fold{i}.pth")
        if os.path.exists(path):
            try:
                m = TransformerV15(input_dim_v15).to(DEVICE)
                m.load_state_dict(torch.load(path, map_location=DEVICE))
                m.eval()
                models_v15.append(m)
            except Exception as e:
                print(f"Error loading V15 Fold {i}: {e}")
            
    print(f"Loaded {len(models_v23)} V23 models and {len(models_v15)} V15 models.")
    return models_v23, models_v15

def inference():
    print("Loading Test Data...")
    test_meta = pd.read_csv(TEST_FILE)
    submission = pd.read_csv(SAMPLE_SUB_FILE)
    submission = submission.merge(test_meta, on="game_episode", how="left")
    
    # Dummy for Input Dim Calculation
    # Note: Use a dummy with all required columns to avoid errors if specific columns are checked
    dummy_df = pd.DataFrame({
        'start_x':[0],'start_y':[0],'end_x':[0],'end_y':[0],
        'time_seconds':[0],'is_home':[0],'type_name':['Pass'],'result_name':['Success']
    })
    # Replicate to ensure window size
    dummy_large = pd.concat([dummy_df]*10, ignore_index=True)
    
    seq23_dummy = extract_features_v23(dummy_large)
    seq15_dummy = extract_features_v15(dummy_large)
    
    print(f"V23 Input Dim: {seq23_dummy.shape[1]}")
    print(f"V15 Input Dim: {seq15_dummy.shape[1]}")
    
    models_v23, models_v15 = load_models(seq23_dummy.shape[1], seq15_dummy.shape[1])
    
    if not models_v23 or not models_v15:
        print("CRITICAL: Failed to load models. Check weights folder.")
        return

    preds_x, preds_y = [], []
    GOAL_X, GOAL_Y = 105.0, 34.0
    SHOT_WEIGHT = 0.3
    
    for _, row in tqdm(submission.iterrows(), total=len(submission), desc="Inference"):
        relative_path = row["path"]
        if relative_path.startswith("./"): relative_path = relative_path[2:]
        full_path = os.path.join(BASE_PATH, relative_path)
        
        try:
            g = pd.read_csv(full_path).reset_index(drop=True)
            
            # --- V23 Prediction (TTA Included) ---
            seq_v23 = extract_features_v23(g)
            x_v23 = torch.tensor(seq_v23).unsqueeze(0).to(DEVICE)
            len_v23 = torch.tensor([seq_v23.shape[0]]).to(DEVICE)
            
            # TTA Flip
            seq_v23_f = seq_v23.copy()
            seq_v23_f[:, 1] = 1.0 - seq_v23_f[:, 1]
            seq_v23_f[:, 3] = -seq_v23_f[:, 3]
            seq_v23_f[:, 5] = -seq_v23_f[:, 5]
            seq_v23_f[:, 8] = -seq_v23_f[:, 8]
            if seq_v23_f.shape[1] > 20: seq_v23_f[:, -2] = -seq_v23_f[:, -2]
            x_v23_f = torch.tensor(seq_v23_f).unsqueeze(0).to(DEVICE)
            
            p23_sum = np.zeros(2)
            for m in models_v23:
                with torch.no_grad():
                    mu1 = m(x_v23, len_v23).cpu().numpy()[0]
                    mu2_f = m(x_v23_f, len_v23).cpu().numpy()[0]
                    mu2 = np.array([mu2_f[0], 1.0 - mu2_f[1]])
                    p23_sum += (mu1 + mu2) / 2.0
            pred_v23 = (p23_sum / len(models_v23)) * [105.0, 68.0]
            
            # --- V15 Prediction (TTA Included) ---
            seq_v15 = extract_features_v15(g)
            x_v15 = torch.tensor(seq_v15).unsqueeze(0).to(DEVICE)
            len_v15 = torch.tensor([seq_v15.shape[0]]).to(DEVICE)
            
            # TTA Flip
            seq_v15_f = seq_v15.copy()
            seq_v15_f[:, 1] = 1.0 - seq_v15_f[:, 1]
            seq_v15_f[:, 3] = -seq_v15_f[:, 3]
            seq_v15_f[:, 5] = -seq_v15_f[:, 5]
            seq_v15_f[:, 8] = -seq_v15_f[:, 8]
            seq_v15_f[:, -2] = -seq_v15_f[:, -2]
            x_v15_f = torch.tensor(seq_v15_f).unsqueeze(0).to(DEVICE)
            
            p15_sum = np.zeros(2)
            for m in models_v15:
                with torch.no_grad():
                    out1 = m(x_v15, len_v15).cpu().numpy()[0]
                    out2_f = m(x_v15_f, len_v15).cpu().numpy()[0]
                    out2 = np.array([out2_f[0], 1.0 - out2_f[1]])
                    p15_sum += (out1 + out2) / 2.0
            pred_v15 = (p15_sum / len(models_v15)) * [105.0, 68.0]
            
            # --- Ensemble 6:4 ---
            final_pred_x = pred_v23[0] * W_V23 + pred_v15[0] * W_V15
            final_pred_y = pred_v23[1] * W_V23 + pred_v15[1] * W_V15
            
            # Shot Heuristic
            last_type = g.iloc[-1]['type_name']
            if isinstance(last_type, str) and 'Shot' in last_type:
                final_pred_x = (final_pred_x * (1 - SHOT_WEIGHT)) + (GOAL_X * SHOT_WEIGHT)
                final_pred_y = (final_pred_y * (1 - SHOT_WEIGHT)) + (GOAL_Y * SHOT_WEIGHT)
                
            preds_x.append(np.clip(final_pred_x, 0, 105))
            preds_y.append(np.clip(final_pred_y, 0, 68))
            
        except Exception as e:
            # print(f"Error on {relative_path}: {e}")
            preds_x.append(52.5)
            preds_y.append(34.0)
            
    sub_df = pd.read_csv(SAMPLE_SUB_FILE)
    sub_df['end_x'] = preds_x
    sub_df['end_y'] = preds_y
    sub_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal Submission Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    inference()