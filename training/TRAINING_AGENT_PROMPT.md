# Neural Network Training Agent Prompt

You are an AI agent tasked with training neural networks for Connect Four (MakeFour). Your goal is to create trained ONNX models that can be deployed to Cloudflare Workers for real-time bot gameplay.

## Context

MakeFour is a Connect Four game with AI bots. The current neural engine (`functions/lib/engines/neural-engine.ts`) supports ONNX model inference but currently falls back to pattern-based heuristics because no trained models exist yet.

## Your Mission

Create a complete training pipeline that produces deployable ONNX models at multiple skill levels.

---

## Part 1: Understanding the Model Interface

### Input Format (Board Encoding)
```
Float32Array[84] - One-hot encoded board state
- Indices 0-41: Player 1 positions (row-major: row*7 + col)
- Indices 42-83: Player 2 positions

Example encoding:
  Row 0, Col 3 occupied by Player 1 → input[3] = 1.0
  Row 5, Col 0 occupied by Player 2 → input[42 + 35] = input[77] = 1.0
  Empty cells → 0.0
```

### Output Format
```typescript
interface ModelOutput {
  policy: number[]  // 7 values, softmax probabilities for each column
  value?: number    // -1.0 (losing) to 1.0 (winning), optional
}
```

### Model Metadata (for registration)
```typescript
interface ModelMetadata {
  id: string                    // e.g., 'cnn-v1-1200elo'
  name: string                  // Human-readable name
  architecture: 'mlp' | 'cnn' | 'transformer'
  expectedElo: number           // Estimated playing strength
  sizeBytes: number             // Model file size
  url: string                   // CDN URL for model file
  version: string               // Semantic version
  encoding: 'onehot-6x7x3' | 'bitboard' | 'flat-binary'
  training?: {
    games: number
    epochs: number
    date: string
  }
}
```

---

## Part 2: Training Data Generation

### Option A: Self-Play with MCTS (Recommended - AlphaZero style)
1. Start with a random/weak policy network
2. Run MCTS simulations using the network for policy/value guidance
3. Play games against itself, storing (state, mcts_policy, outcome) tuples
4. Train network on collected data
5. Repeat with improved network

### Option B: Teacher Distillation
1. Use the existing minimax engine as a teacher
2. For each position, run minimax to depth 10+ to get "correct" moves
3. Store (state, minimax_policy, minimax_value) tuples
4. Train network to match minimax outputs

### Option C: Solved Positions Database
1. Connect Four is solved - use a database of perfect play
2. Download/generate positions with known outcomes
3. Train network on (state, perfect_move, outcome) tuples

### Recommended Approach: Hybrid
1. Generate 100K+ positions using self-play MCTS
2. Augment with 50K+ teacher-distilled positions from minimax
3. Add 10K solved opening positions for accuracy
4. Data augmentation: Mirror boards horizontally (column 0↔6, 1↔5, etc.)

---

## Part 3: Model Architectures

### Target Models (create all three)

#### 1. MLP Model (~950 ELO) - "Spark"
```python
# Simple feedforward network for the beginner neural bot
Input: 84 → Dense(256, ReLU) → Dense(128, ReLU) →
       Policy Head: Dense(7, Softmax)
       Value Head: Dense(1, Tanh)

Target size: < 100KB ONNX
Training: 50K games, distilled from depth-2 minimax
```

#### 2. CNN Model (~1400 ELO) - "Synapse"
```python
# Convolutional network that sees spatial patterns
Input: Reshape to (2, 6, 7) channels
Conv2D(32, 3x3, padding='same', ReLU) →
Conv2D(64, 3x3, padding='same', ReLU) →
Flatten → Dense(128, ReLU) →
Policy Head: Dense(7, Softmax)
Value Head: Dense(1, Tanh)

Target size: < 500KB ONNX
Training: 200K games self-play + teacher distillation
```

#### 3. ResNet Model (~1650 ELO) - "Cortex"
```python
# Residual network for stronger pattern recognition
Input: Reshape to (2, 6, 7) channels
Conv2D(64, 3x3, padding='same') → BatchNorm → ReLU →
ResBlock(64) × 4 →  # Each block: Conv→BN→ReLU→Conv→BN→Skip→ReLU
Global Average Pool → Dense(128, ReLU) →
Policy Head: Dense(7, Softmax)
Value Head: Dense(1, Tanh)

Target size: < 1MB ONNX
Training: 500K games self-play with MCTS (800 simulations)
```

---

## Part 4: Training Pipeline

### Directory Structure
```
training/
├── TRAINING_AGENT_PROMPT.md  (this file)
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── generator.py      # Self-play data generation
│   │   ├── teacher.py        # Minimax teacher distillation
│   │   └── augment.py        # Data augmentation
│   ├── models/
│   │   ├── mlp.py            # MLP architecture
│   │   ├── cnn.py            # CNN architecture
│   │   └── resnet.py         # ResNet architecture
│   ├── train.py              # Training loop
│   ├── evaluate.py           # Play against minimax to estimate ELO
│   └── export.py             # Export to ONNX
├── data/                     # Generated training data
├── checkpoints/              # Training checkpoints
└── models/                   # Final ONNX models
```

### Training Steps

1. **Setup Environment**
```bash
cd training
python -m venv venv
source venv/bin/activate
pip install torch numpy onnx onnxruntime tqdm
```

2. **Generate Training Data**
```bash
# Generate self-play games
python src/data/generator.py --games 100000 --output data/selfplay.npz

# Generate teacher data from minimax
python src/data/teacher.py --positions 50000 --depth 8 --output data/teacher.npz
```

3. **Train Models**
```bash
# Train MLP (Spark)
python src/train.py --arch mlp --data data/teacher.npz --epochs 100 --output checkpoints/mlp

# Train CNN (Synapse)
python src/train.py --arch cnn --data data/selfplay.npz,data/teacher.npz --epochs 200 --output checkpoints/cnn

# Train ResNet (Cortex)
python src/train.py --arch resnet --data data/selfplay.npz --epochs 300 --mcts-guided --output checkpoints/resnet
```

4. **Evaluate Models**
```bash
# Play against minimax at various depths to estimate ELO
python src/evaluate.py --model checkpoints/mlp/best.pt --games 200
python src/evaluate.py --model checkpoints/cnn/best.pt --games 200
python src/evaluate.py --model checkpoints/resnet/best.pt --games 200
```

5. **Export to ONNX**
```bash
python src/export.py --model checkpoints/mlp/best.pt --output models/mlp-spark-v1.onnx
python src/export.py --model checkpoints/cnn/best.pt --output models/cnn-synapse-v1.onnx
python src/export.py --model checkpoints/resnet/best.pt --output models/resnet-cortex-v1.onnx
```

---

## Part 5: ONNX Export Requirements

### Export Code Template
```python
import torch
import torch.onnx

def export_to_onnx(model, output_path):
    model.eval()

    # Dummy input matching expected shape
    dummy_input = torch.zeros(1, 84)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['board_input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'board_input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )

    # Verify the model
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    print(f"Exported to {output_path}")
```

### Validation
After export, validate in JavaScript environment:
```javascript
// Test in Node.js with onnxruntime-node
const ort = require('onnxruntime-node');

async function testModel(modelPath) {
  const session = await ort.InferenceSession.create(modelPath);

  // Create test input (empty board)
  const input = new Float32Array(84).fill(0);
  const tensor = new ort.Tensor('float32', input, [1, 84]);

  const results = await session.run({ board_input: tensor });
  console.log('Policy:', results.policy.data);
  console.log('Value:', results.value?.data);
}
```

---

## Part 6: Integration with MakeFour

### Update MODEL_REGISTRY
After training, update `functions/lib/engines/neural-engine.ts`:

```typescript
export const MODEL_REGISTRY: ModelMetadata[] = [
  {
    id: 'heuristic-v1',
    name: 'Heuristic Baseline',
    architecture: 'mlp',
    expectedElo: 1200,
    sizeBytes: 0,
    url: '',
    version: '1.0.0',
    encoding: 'flat-binary',
  },
  {
    id: 'mlp-spark-v1',
    name: 'Spark Neural (MLP)',
    architecture: 'mlp',
    expectedElo: 950,
    sizeBytes: 98304,  // ~96KB
    url: 'https://cdn.makefour.org/models/mlp-spark-v1.onnx',
    version: '1.0.0',
    encoding: 'flat-binary',
    training: {
      games: 50000,
      epochs: 100,
      date: '2024-12-19',
    },
  },
  {
    id: 'cnn-synapse-v1',
    name: 'Synapse Neural (CNN)',
    architecture: 'cnn',
    expectedElo: 1400,
    sizeBytes: 491520,  // ~480KB
    url: 'https://cdn.makefour.org/models/cnn-synapse-v1.onnx',
    version: '1.0.0',
    encoding: 'onehot-6x7x3',
    training: {
      games: 200000,
      epochs: 200,
      date: '2024-12-19',
    },
  },
  {
    id: 'resnet-cortex-v1',
    name: 'Cortex Neural (ResNet)',
    architecture: 'cnn',
    expectedElo: 1650,
    sizeBytes: 1048576,  // ~1MB
    url: 'https://cdn.makefour.org/models/resnet-cortex-v1.onnx',
    version: '1.0.0',
    encoding: 'onehot-6x7x3',
    training: {
      games: 500000,
      epochs: 300,
      date: '2024-12-19',
    },
  },
]
```

### Update Bot Personas in schema.sql
Update the neural bots to use real model IDs:
```sql
UPDATE bot_personas SET ai_config = '{"modelId":"mlp-spark-v1","temperature":0.8,"useHybridSearch":false}' WHERE id = 'spark';
UPDATE bot_personas SET ai_config = '{"modelId":"cnn-synapse-v1","temperature":0.4,"useHybridSearch":true,"hybridDepth":4}' WHERE id = 'synapse';
UPDATE bot_personas SET ai_config = '{"modelId":"resnet-cortex-v1","temperature":0.1,"useHybridSearch":true,"hybridDepth":6}' WHERE id = 'cortex';
```

---

## Part 7: Deployment

### Model Hosting Options
1. **Cloudflare R2** - Native integration with Workers
2. **GitHub Releases** - Free hosting for model files
3. **CDN (jsDelivr/unpkg)** - If models are in npm package

### Implement ONNXInference Class
The neural engine needs a real ONNX inference implementation:

```typescript
// In neural-engine.ts, implement ONNXInference class
class ONNXInference implements ModelInference {
  private session: any = null
  private modelId: string

  constructor(modelId: string) {
    this.modelId = modelId
  }

  async load(): Promise<void> {
    const metadata = getModelMetadata(this.modelId)
    if (!metadata) throw new Error(`Unknown model: ${this.modelId}`)

    // Fetch model from CDN
    const response = await fetch(metadata.url)
    const modelBuffer = await response.arrayBuffer()

    // Load with ONNX Runtime (need to add onnxruntime-web dependency)
    const ort = await import('onnxruntime-web')
    this.session = await ort.InferenceSession.create(modelBuffer)
  }

  isLoaded(): boolean {
    return this.session !== null
  }

  async predict(boardInput: Float32Array): Promise<ModelOutput> {
    const ort = await import('onnxruntime-web')
    const tensor = new ort.Tensor('float32', boardInput, [1, 84])
    const results = await this.session.run({ board_input: tensor })

    return {
      policy: Array.from(results.policy.data as Float32Array),
      value: results.value ? (results.value.data as Float32Array)[0] : undefined
    }
  }
}
```

---

## Deliverables Checklist

- [ ] Training data generation scripts
- [ ] MLP model architecture and training
- [ ] CNN model architecture and training
- [ ] ResNet model architecture and training
- [ ] ONNX export and validation
- [ ] ELO evaluation against minimax
- [ ] Model files uploaded to CDN
- [ ] MODEL_REGISTRY updated with new models
- [ ] ONNXInference class implemented
- [ ] Bot personas updated with real modelIds
- [ ] End-to-end testing of neural bots in production

---

## Success Criteria

1. **Spark (MLP)** plays at ~950 ELO, beats depth-2 minimax 70%+
2. **Synapse (CNN)** plays at ~1400 ELO, competitive with depth-5 minimax
3. **Cortex (ResNet)** plays at ~1650 ELO, competitive with depth-7 minimax
4. All models load and run inference in < 50ms on Cloudflare Workers
5. Model files are < 1MB each for fast CDN delivery

Good luck, training agent!
