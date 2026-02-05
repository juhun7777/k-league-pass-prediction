<p align="center">
  <img src="assets/logo.jpg" alt="K-League Pass Prediction" width="100%"/>
</p>

<h1 align="center">K-League Pass Destination Prediction</h1>

<p align="center">
  <strong>Hybrid Transformer Ensemble for Tactical Pass Coordinate Prediction in K-League Football</strong>
</p>

<p align="center">
  <em>Reading Tactical Intent Beyond Data: A Physics-Informed Deep Learning Approach</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transformer-Encoder-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/CUDA-Supported-76B900?style=for-the-badge&logo=nvidia&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🏆_Rank-13th_Place-gold?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/🎖️_Award-Encouragement_Prize-orange?style=for-the-badge"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Competition-DACON%20K--League%20AI-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Task-Coordinate%20Regression-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

---

## Overview

In modern football, victory is determined not merely by individual player skills, but by the ability to create invisible spaces and orchestrate organic team patterns. While every movement on the pitch is now recorded as data, extracting the **tactical intent** that changes the flow of a match remains a significant challenge.

This project develops an AI model that predicts the optimal destination coordinates (X, Y) of the final pass in a given play sequence, learning to understand match situations like an experienced player and anticipate the next move.

> **Key Innovation**: Combining deterministic stability with probabilistic precision through a hybrid ensemble of two complementary Transformer architectures.

> 🏆 **Achievement**: This solution achieved **13th place** and received the **Encouragement Prize (장려상)** at the DACON K-League AI Challenge.

---

## Competition Information

| Item | Description |
|:-----|:------------|
| **Competition** | DACON K-League Pass Coordinate Prediction AI Challenge |
| **Organizer** | K-League & DACON |
| **Period** | 2025.12.01 ~ 2026.01.12 |
| **Task** | Predict final pass destination coordinates (X, Y) from play sequences |
| **Metric** | Euclidean Distance (Mean) |
| **Coordinate System** | FIFA Standard Pitch (105m × 68m), Left-to-Right Attack Direction |
| **Final Rank** | 🏆 **13th Place** |
| **Award** | 🎖️ **Encouragement Prize (장려상)** |

---

## Problem Definition

Given a sequence of football events $\mathcal{S} = \{e_1, e_2, ..., e_n\}$, predict the destination coordinates of the final pass:

$$\hat{y} = f_\theta(\mathcal{S}) = (\hat{x}, \hat{y}) \in [0, 105] \times [0, 68]$$

**Evaluation Metric:**
$$\text{Score} = \frac{1}{N} \sum_{i=1}^{N} \sqrt{(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2}$$

---

## Model Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Model Architecture" width="800"/>
</p>

### Hybrid Ensemble Strategy

Our solution employs two complementary models with distinct roles:

| Model | Role | Window | Loss Function | Weight |
|:------|:-----|:------:|:--------------|:------:|
| **V15** | Stability Anchor | 3 | Euclidean Distance | 0.4 |
| **V23** | Precision Expert | 5 | Bivariate Gaussian NLL + L2 | 0.6 |

### Model V15: Stability Anchor

```
Input → Linear(D, 128) → PositionalEncoding → TransformerEncoder(3L, 4H) → FC → (x, y)
```

- **Focus**: Recent 3 events for immediate reaction learning
- **Output**: Direct coordinate regression (2D)
- **Role**: Provides geometric centroid estimation, minimizing average risk

### Model V23: Precision Expert

```
Input → Linear(D, 128) → PositionalEncoding → TransformerEncoder(3L, 4H) → FC → (μ, σ, ρ)
```

- **Focus**: Recent 5 events with 34+ event types for build-up context
- **Output**: Bivariate Gaussian parameters (μx, μy, σx, σy, ρ)
- **Key Technique**: Correlation coefficient (ρ) learning for non-linear trajectories (diagonal/curved passes)
- **Residual Learning**: Predicts `start_pos + Δ` for training stability

---

## Feature Engineering

### Kinematic Features
| Feature | Description | Normalization |
|:--------|:------------|:--------------|
| `sx, sy` | Start coordinates | ÷ (105, 68) |
| `dx, dy` | Displacement vector | Raw |
| `dist` | Movement distance | $\sqrt{dx^2 + dy^2}$ |
| `angle` | Movement angle | $\arctan2(dy, dx)$ |
| `vx, vy, speed` | Velocity components | ÷ time_delta |

### Goal Context Features
| Feature | Formula | Purpose |
|:--------|:--------|:--------|
| `goal_dist` | $\sqrt{(105-x)^2 + (34-y)^2}$ | Attack purposefulness |
| `goal_angle` | $\arctan2(34-y, 105-x)$ | Shooting angle awareness |
| `center_dist` | $\sqrt{(52.5-x)^2 + (34-y)^2}$ | Positional context |

### Categorical Features
- **Event Type**: One-hot encoded (34 types for V23, 15 types for V15)
- **Result**: One-hot encoded (Successful, Unsuccessful, etc.)
- **Team Context**: `is_home` binary flag

---

## Dataset

### Data Schema

| Column | Type | Description |
|:-------|:-----|:------------|
| `game_episode` | string | Unique identifier `{game_id}_{episode_id}` |
| `period_id` | int | Match period (1: First half, 2: Second half) |
| `time_seconds` | float | Elapsed time within period |
| `team_id` | int | Team identifier |
| `player_id` | float | Player identifier |
| `type_name` | string | Event type (Pass, Carry, Shot, etc.) |
| `result_name` | string | Event result (Successful, Unsuccessful) |
| `start_x, start_y` | float | Event start coordinates |
| `end_x, end_y` | float | Event end coordinates (**Target**) |
| `is_home` | bool | Home team indicator |

### Event Types (62 Categories)

<details>
<summary>Click to expand event type list</summary>

| Type | Results | Description |
|:-----|:--------|:------------|
| Pass | Successful, Unsuccessful | Standard pass between teammates |
| Carry | - | Ball movement while dribbling |
| Shot | Goal, On Target, Off Target, Blocked | Shot attempt |
| Cross | Successful, Unsuccessful | Wide-area pass to center |
| Duel | Successful, Unsuccessful | 1v1 challenge |
| Tackle | Successful, Unsuccessful | Defensive tackle |
| Interception | - | Intercepting opponent's pass |
| Clearance | - | Defensive clearance |
| Goal Kick | Successful, Unsuccessful | Goalkeeper distribution |
| Throw-In | Successful, Unsuccessful | Throw-in restart |
| Pass_Corner | Successful, Unsuccessful | Corner kick delivery |
| Pass_Freekick | Successful, Unsuccessful | Free kick delivery |
| ... | ... | ... |

</details>

---

## Training Strategy

### Data Augmentation: Physics Mirroring

Exploiting the symmetric nature of football pitches:

```python
# Y-axis reflection (pitch symmetry)
seq_aug[:, 1] = 1.0 - seq_aug[:, 1]   # sy → 1 - sy
seq_aug[:, 3] = -seq_aug[:, 3]         # dy → -dy
seq_aug[:, 5] = -seq_aug[:, 5]         # angle → -angle
seq_aug[:, 8] = -seq_aug[:, 8]         # vy → -vy
target[1] = 1.0 - target[1]            # target_y → 1 - target_y
```

**Effect**: 2× data augmentation while preserving physical consistency

### Cross-Validation: Group K-Fold

```python
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=episode_ids)):
    # Episode-based splitting prevents temporal leakage
```

### Training Configuration

| Hyperparameter | V15 | V23 |
|:---------------|:----|:----|
| Batch Size | 64 | 64 |
| Learning Rate | 1e-3 | 1e-3 |
| Epochs | 100 | 100 |
| Optimizer | Adam | Adam |
| Scheduler | ReduceLROnPlateau | ReduceLROnPlateau |
| d_model | 128 | 128 |
| n_heads | 4 | 4 |
| n_layers | 3 | 3 |
| Dropout | 0.123 | 0.15 |

---

## Inference Optimization

### Test-Time Augmentation (TTA)

```python
# Original prediction
pred_orig = model(x_original)

# Flipped prediction
pred_flip = model(x_flipped)
pred_flip[1] = 1.0 - pred_flip[1]  # Reverse Y transformation

# Average to cancel positional bias
final_pred = (pred_orig + pred_flip) / 2
```

### Shot Heuristic (Domain Knowledge Injection)

```python
if last_event_type == 'Shot':
    GOAL_X, GOAL_Y = 105.0, 34.0
    SHOT_WEIGHT = 0.3
    pred_x = pred_x * (1 - SHOT_WEIGHT) + GOAL_X * SHOT_WEIGHT
    pred_y = pred_y * (1 - SHOT_WEIGHT) + GOAL_Y * SHOT_WEIGHT
```

### Final Ensemble

$$\hat{y}_{final} = 0.6 \cdot \hat{y}_{V23} + 0.4 \cdot \hat{y}_{V15}$$

---

## Project Structure

```
k-league-pass-prediction/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── train_v23.py             # V23 Bivariate Gaussian model training
├── train_v15.py             # V15 Euclidean regression model training
├── inference.py             # Ensemble inference with TTA
├── assets/
│   ├── logo.jpg             # Project logo
│   └── architecture.png     # Model architecture diagram
├── weights/
│   ├── v23_fold1.pth        # V23 fold 1 weights
│   ├── v23_fold2.pth        # V23 fold 2 weights
│   ├── v23_fold3.pth        # V23 fold 3 weights
│   ├── v23_fold4.pth        # V23 fold 4 weights
│   ├── v23_fold5.pth        # V23 fold 5 weights
│   ├── v15_fold1.pth        # V15 fold 1 weights
│   ├── v15_fold2.pth        # V15 fold 2 weights
│   ├── v15_fold3.pth        # V15 fold 3 weights
│   ├── v15_fold4.pth        # V15 fold 4 weights
│   └── v15_fold5.pth        # V15 fold 5 weights
├── train.csv                # Training data (not included)
├── test.csv                 # Test metadata (not included)
└── test/                    # Test episode files (not included)
```

---

## Installation

### Requirements

- Python 3.12+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/juhun7777/k-league-pass-prediction.git
cd k-league-pass-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.1
torch==2.8.0
tqdm==4.66.5
pyarrow==16.1.0
joblib==1.4.2
```

---

## Usage

### Training

Pre-trained weights are provided in the `weights/` directory. To retrain:

```bash
# Train V23 model (Bivariate Gaussian)
python train_v23.py

# Train V15 model (Euclidean)
python train_v15.py
```

### Inference

```bash
# Generate submission file
python inference.py
```

**Output**: `submission.csv` with predicted `end_x`, `end_y` coordinates

---

## Results

### Competition Performance

| Metric | Score |
|:-------|:-----:|
| **Final Rank** | 🏆 **13th / 700+ Teams** |
| **Award** | 🎖️ **Encouragement Prize (장려상)** |
| **Private Score** | Euclidean Distance |

### Cross-Validation Performance

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|:------|:------:|:------:|:------:|:------:|:------:|:----:|
| V15 | - | - | - | - | - | - |
| V23 | - | - | - | - | - | - |
| **Ensemble** | - | - | - | - | - | **-** |

### Ablation Study

| Configuration | Validation Distance |
|:--------------|:-------------------:|
| V15 only | - |
| V23 only | - |
| Ensemble (0.5:0.5) | - |
| Ensemble (0.6:0.4) | - |
| + TTA | - |
| + Shot Heuristic | - |

---

## Key Insights

1. **Complementary Model Design**: V15 provides stable centroid estimation while V23 captures complex distributional patterns

2. **Bivariate Gaussian Benefits**: Learning correlation coefficient (ρ) enables prediction of non-linear trajectories common in football (diagonal passes, through balls)

3. **Physics-Informed Augmentation**: Y-axis mirroring respects football's symmetric nature, doubling effective training data

4. **Domain Knowledge Integration**: Shot heuristic injects tactical knowledge that pure data-driven models may miss

5. **Residual Prediction**: V23's `start_pos + Δ` formulation stabilizes training by reducing output variance

---

## Development Environment

| Component | Specification |
|:----------|:--------------|
| OS | Windows 11 |
| Python | 3.12.7 |
| PyTorch | 2.8.0 |
| CUDA | 12.x |
| GPU | NVIDIA RTX Series (Recommended) |

---

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. Fernández, J., & Bornn, L. (2018). "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." *MIT Sloan Sports Analytics Conference*.
3. Decroos, T., et al. (2019). "Actions Speak Louder than Goals: Valuing Player Actions in Soccer." *KDD*.

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{kleague-pass-prediction-2025,
  author       = {JH_99},
  title        = {K-League Pass Destination Prediction: Hybrid Transformer Ensemble},
  year         = {2025},
  publisher    = {GitHub},
  note         = {13th Place, Encouragement Prize at DACON K-League AI Challenge},
  howpublished = {\url{https://github.com/juhun7777/k-league-pass-prediction}}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaborations:

- **GitHub**: [@juhun7777](https://github.com/juhun7777)

---

<p align="center">
  <strong>Reading Tactical Intent Beyond Data</strong><br>
  <em>Transforming football analytics from event recording to tactical understanding</em>
</p>
