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

## Directory Structure

```
training/
├── src/
│   └── data/
│       ├── encoding.py      # Position encoding (matches TypeScript)
│       ├── game.py          # Connect Four game logic
│       ├── dataset.py       # PyTorch dataset classes
│       ├── export.py        # Data export utilities
│       └── validation.py    # Data validation
├── tests/
│   ├── test_encoding.py     # Encoding parity tests
│   ├── test_game.py         # Game logic tests
│   ├── test_dataset.py      # Dataset tests
│   └── test_validation.py   # Validation tests
├── data/
│   ├── games/               # Raw game records
│   └── processed/           # Processed tensors
├── configs/                 # Training configs
└── scripts/                 # CLI scripts
```

## Related Issues

- Epic: #81 - Neural Network Training Pipeline
- Model Architectures: #83
- Training Infrastructure: #84
