# MakeFour Neural Network Training

Training pipeline for Connect Four neural networks with adversarial self-play.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"
```

## Quick Start

### Generate Synthetic Training Data

```python
from src.data import generate_synthetic_dataset, save_games_jsonl

# Generate 1000 random games
games = generate_synthetic_dataset(num_games=1000, seed=42)
save_games_jsonl(games, "data/games/synthetic/random_1000.jsonl")

# Generate with center-column bias (more realistic)
biased_games = generate_biased_dataset(num_games=1000, seed=42)
save_games_jsonl(biased_games, "data/games/synthetic/biased_1000.jsonl")
```

### Load Data for Training

```python
from src.data import ConnectFourDataset, load_games_jsonl
from torch.utils.data import DataLoader

# Load from JSONL
games = load_games_jsonl("data/games/synthetic/random_1000.jsonl")

# Create PyTorch dataset
dataset = ConnectFourDataset(
    games=games,
    encoding="flat-binary",  # or "onehot-6x7x3" for CNNs
    augment=True,  # Horizontal flip augmentation
)

# Create dataloader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

for batch in loader:
    board = batch["board"]      # [64, 85] for flat-binary
    move = batch["move"]        # [64] target column
    value = batch["value"]      # [64] game outcome
    legal_mask = batch["legal_mask"]  # [64, 7] valid moves
```

### Convert MakeFour API Export

```python
from src.data import convert_api_export, save_games_jsonl
import json

# Load API export JSON
with open("makefour_export.json") as f:
    api_data = json.load(f)

# Convert to training format
games = convert_api_export(api_data)
save_games_jsonl(games, "data/games/human/export.jsonl")
```

### Validate Data

```python
from src.data import validate_dataset, get_dataset_statistics

# Validate all games
validation = validate_dataset(games)
print(f"Valid: {validation['valid_games']}/{validation['total_games']}")

# Get statistics
stats = get_dataset_statistics(games)
print(f"Total positions: {stats['total_positions']}")
print(f"Outcomes: {stats['outcomes']}")
```

## Data Format

### Game Record (JSONL)

Each line is a JSON object with this structure:

```json
{
  "game_id": "uuid-string",
  "positions": [
    {
      "board": [[null,null,...], ...],
      "to_move": 1,
      "move_played": 3,
      "result": 1.0,
      "moves_to_end": 12,
      "policy_target": null,
      "value_target": null
    }
  ],
  "metadata": {
    "source": "api_export",
    "player_rating": 1200
  }
}
```

### Encoding Types

| Type | Shape | Description |
|------|-------|-------------|
| `flat-binary` | `[85]` | 42 P1 bits + 42 P2 bits + 1 current player |
| `onehot-6x7x3` | `[126]` | 3 channels (P1, P2, current player) flattened |
| `bitboard` | `[4]` | P1 bits, P2 bits, player bit, move count |

All encodings match the TypeScript implementation in `src/ai/neural/encoding.ts`.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_encoding.py -v
```

## Training

### Quick Training Example

```python
from src.data import ConnectFourDataset, load_games_jsonl, create_train_val_test_split
from src.training import (
    Trainer, TrainerConfig, Checkpoint, ConsoleLogger, EarlyStopping,
    create_optimizer, create_scheduler
)
from torch.utils.data import DataLoader
import torch.nn as nn

# Load data
games = load_games_jsonl("data/games/synthetic/random_1000.jsonl")
train_games, val_games, _ = create_train_val_test_split(games)

train_dataset = ConnectFourDataset(games=train_games, encoding="flat-binary", augment=True)
val_dataset = ConnectFourDataset(games=val_games, encoding="flat-binary", augment=False)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Create model (your model with policy and value heads)
model = YourDualHeadModel()

# Create optimizer and scheduler
optimizer = create_optimizer(model.parameters(), optimizer_type="adamw", learning_rate=0.001)
scheduler = create_scheduler(optimizer, scheduler_type="cosine", epochs=100)

# Create trainer with callbacks
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    config=TrainerConfig(epochs=100, batch_size=256, device="auto"),
    callbacks=[
        ConsoleLogger(log_every=100),
        Checkpoint(checkpoint_dir="./checkpoints", save_best=True),
        EarlyStopping(patience=10, metric="val_loss"),
    ],
)

# Train
results = trainer.train(train_loader, val_loader)
```

### Training Script CLI

```bash
# Basic training
python -m scripts.train --config configs/train/supervised-mlp-tiny.yaml --data ./data/games.jsonl

# With custom output directory
python -m scripts.train -c configs/train/supervised-mlp-small.yaml -d ./data/games.jsonl -o ./my_checkpoints

# Override config values
python -m scripts.train -c configs/train/supervised-mlp-tiny.yaml -d ./data/games.jsonl --epochs 50 --lr 0.0001

# Resume from checkpoint
python -m scripts.train -c configs/train/supervised-mlp-tiny.yaml -d ./data/games.jsonl --resume ./checkpoints/checkpoint_epoch_10.pt
```

### Training Configuration

Training configs are YAML files in `configs/train/`. Example:

```yaml
model:
  type: mlp-tiny
  dropout: 0.1

training:
  epochs: 100
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  policy_weight: 1.0
  value_weight: 0.5
  early_stopping:
    patience: 15
    metric: val_loss
    mode: min

optimizer:
  type: adamw

scheduler:
  type: cosine
  warmup_epochs: 5
  min_lr: 0.000001

data:
  encoding: flat-binary
  augment: true
  train_ratio: 0.8
  val_ratio: 0.1
```

Available configurations:
- `supervised-mlp-tiny.yaml` - Fast training for experiments
- `supervised-mlp-small.yaml` - Balanced performance
- `supervised-mlp-medium.yaml` - Best MLP performance
- `test-quick.yaml` - Quick pipeline validation

## Directory Structure

```
training/
├── src/
│   ├── data/
│   │   ├── encoding.py      # Position encoding (matches TypeScript)
│   │   ├── game.py          # Connect Four game logic
│   │   ├── dataset.py       # PyTorch dataset classes
│   │   ├── export.py        # Data export utilities
│   │   └── validation.py    # Data validation
│   └── training/
│       ├── losses.py        # Loss functions (policy, value, combined)
│       ├── trainer.py       # Main Trainer class
│       ├── optimizers.py    # Optimizer and scheduler factories
│       └── callbacks.py     # Training callbacks (logging, checkpointing)
├── tests/
│   ├── test_encoding.py     # Encoding parity tests
│   ├── test_game.py         # Game logic tests
│   ├── test_dataset.py      # Dataset tests
│   ├── test_validation.py   # Validation tests
│   ├── test_losses.py       # Loss function tests
│   ├── test_trainer.py      # Trainer tests
│   ├── test_callbacks.py    # Callback tests
│   └── test_optimizers.py   # Optimizer/scheduler tests
├── configs/
│   └── train/               # Training configurations
├── scripts/
│   └── train.py             # CLI training script
└── data/
    ├── games/               # Raw game records
    └── processed/           # Processed tensors
```

## Related Issues

- Epic: #81 - Neural Network Training Pipeline
- Model Architectures: #83
- Training Infrastructure: #84
