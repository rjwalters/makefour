# MakeFour Research Framing

A structured research program connecting:
- **Connect Four** (MakeFour environment)
- **Brain-size–performance scaling**
- **ELO as a cognitive capacity curve**
- **Evolutionary constraints ("spider intelligence")**
- **Morphology-inspired neural architectures**

---

## Core Hypothesis

**"Intelligent behavior emerges not only from the number of neurons, but from architecture, embodiment constraints, and environment structure. Therefore, small neural controllers—under the right constraints—can achieve disproportionately high performance in structured domains like Connect Four."**

This parallels the biological observation:

> **Spiders do more with less** — because their *morphology* offloads computation from their neurons.

---

## Research Questions

### RQ1: What is the smallest artificial brain that can play Connect Four competently?

A **minimum viable intelligence** question in a fully solved domain.

**Brain size metrics:**
- Number of parameters
- Number of neurons
- Connectivity pattern / sparsity
- Number of computational steps
- Memory footprint
- Energy use (if using SNNs or neuromorphic simulation)

**Performance metrics:**
- Win rate
- ELO rating
- Distance-to-optimal move selection
- Depth-to-blunder
- Robustness under perturbation

This produces a scaling curve: **Brain size → Performance**

---

### RQ2: How does Connect Four ELO scale with neural network size and architecture constraints?

The "learning curve of constrained intelligence."

**ELO measurement:**
- Self-play plus rating pool of known agents (perfect oracle, heuristic bots, random bots)

**Architectural families to test:**
- Tiny MLPs
- CNNs with local receptive fields
- Shallow RNNs
- SNNs (Spiking Neural Networks)
- Explicit morphology-inspired wiring patterns
- Attention-based microscopic transformers (e.g., 4 heads, 64 params)

**Expected findings:** Not a smooth line, but **phase transitions** where heuristics "snap into place":
- Center-bias
- Column parity
- Immediate win recognition
- Threat trees
- Diagonal heuristics

These emergent heuristics reflect cognitive strategies—*compressed policies*—mirroring how animals with small nervous systems solve ecological tasks.

---

### RQ3: Why do "small brains" perform disproportionately well? (The Spider Hypothesis)

**Spiders exhibit complex behavior despite tiny brains because:**

1. Their **morphology shapes perception and action**
2. The **environment itself stabilizes computation**
3. Their **behaviors are structured into efficient motifs**
4. They exploit **low-dimensional latent structure** in their tasks

**In MakeFour, similar forces exist:**

#### (1) Morphological constraints → architectural inductive biases
- Convolutional structure
- Local receptive fields
- Symmetry tying
- Restricted connectivity (column-limited, neighbor-limited)

#### (2) Environmental structure → simplified representation
Connect Four has:
- Gravity
- Discrete columns
- Strong center bias
- Local interaction patterns
- Limited branching factor
- Highly predictable state transitions

This lets small brains learn strong heuristics *without* complex memory.

#### (3) Action patterns are stereotyped
Move sequences compress well:
- "threat → block → double threat → win"
- "center control → parity play → flank collapse"

These are **behavioral motifs**, like spider web-building patterns.

#### (4) The oracle supplies perfect feedback
Unlike real evolution, you can accelerate "cognitive evolution" with accurate supervision or RL rewards.

---

## Testable Predictions

### Prediction 1
Small networks (< 2k parameters) can achieve ~1200 ELO equivalent with appropriate architectural priors (CNN, local wiring).

### Prediction 2
Architectural constraints matter more than size.
E.g., a 512-parameter CNN beats a 2k-parameter dense MLP.

### Prediction 3
Performance plateaus at certain cognitive "milestones":
- ~1000 ELO → learns center-bias
- ~1300 ELO → learns vertical/horizontal threat detection
- ~1500 ELO → learns diagonal threats
- ~1700 ELO → learns multi-step forcing sequences
- ~2000 ELO → near-perfect tactical avoidance

These plateaus become evidence of *cognitive phases*.

### Prediction 4
A perfectly optimal "oracle" agent is reachable with surprisingly few neurons once forced into an appropriate morphology (e.g., a transformer with tied weights).

### Prediction 5
Generalization to new board sizes is possible only when representation is structured (CNN morphology).

This parallels biological generalization in spiders and insects.

---

## MakeFour as the Testbed

The platform provides:

1. **Solved environment with perfect oracle**
2. **Stable rating system (ELO/Glicko) for cognitive performance**
3. **Replayable games + move-level analysis**
4. **Hook for training agents with constrained architecture**
5. **Mechanism to measure scaling laws cleanly**
6. **Full control over environment complexity**

This is *exactly* what biology lacks: a controlled, reproducible evolutionary landscape.

---

## Overarching Thesis

> **Intelligence is a function of environmental structure + morphological priors + minimal computation.**
> This can be studied precisely in MakeFour, where the cognitive problem is fully observable, perfectly solvable, evolutionarily tractable, and amenable to architectural constraints.

---

## Potential Publication Narratives

- "Scaling Laws for Constrained Intelligence in Solved Environments"
- "How Much Brain Does Connect Four Require?"
- "Morphological Computation in Tiny Neural Controllers"
- "The Spider Hypothesis in Artificial Agents"
- "Emergent Heuristics in Minimal Neural Systems"

**Ultimate deliverable:** A clear, quantitative curve relating *brain size* to *intelligence level* (ELO) in a discrete strategy domain.

---

## Next Steps

Possible directions to develop:
- [ ] Paper abstract
- [ ] Full research proposal
- [ ] Landing page for makefour.org explaining the science
- [ ] Visual diagram of brain-size vs ELO concept
- [ ] Naming scheme for artificial brains (M4-Tiny, M4-Spider, M4-Ant, M4-Frog, etc.)
