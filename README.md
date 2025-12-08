# MakeFour

A Connect Four research platform for studying intelligence scaling in solved environments.

## The Research Question

**How much brain does Connect Four require?**

MakeFour is a testbed for studying the relationship between neural network size, architecture, and cognitive performance in a fully solved game environment. Connect Four is uniquely suited for this research:

- **Fully solved** — a perfect oracle exists for ground truth
- **Structured** — gravity, discrete columns, local interactions
- **Measurable** — ELO provides a continuous cognitive capacity metric
- **Tractable** — small enough to train thousands of agents

### The Spider Hypothesis

> Spiders exhibit complex behavior despite tiny brains because their morphology offloads computation.

We apply this insight to artificial agents: **architectural constraints (morphology) can matter more than raw parameter count**. A 512-parameter CNN may outperform a 2k-parameter MLP because its structure matches the problem's geometry.

### Research Goals

1. **Minimum viable intelligence** — Find the smallest networks that achieve specific ELO thresholds
2. **Scaling curves** — Map brain size → performance across architectural families
3. **Cognitive phase transitions** — Identify ELO plateaus where specific heuristics emerge
4. **Morphological computation** — Quantify how architecture substitutes for size

See [`documents/research-framing.md`](./documents/research-framing.md) for the full research program.

## Tech Stack

- **Frontend**: React + TypeScript + Vite
- **Styling**: Tailwind CSS + shadcn/ui
- **Backend**: Cloudflare Pages + Workers
- **Database**: Cloudflare D1 (SQLite)
- **Game Engine**: TypeScript Connect Four implementation with perfect play oracle (planned)

## Getting Started

### Prerequisites

- Node.js 18+
- npm or pnpm
- Cloudflare account (for deployment)
- Wrangler CLI (`npm install -g wrangler`)

### Installation

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

### Database Setup

```bash
# Login to Cloudflare
npm run wrangler:login

# Create D1 database
npm run db:create

# Update database_id in wrangler.toml, then:
npm run db:migrate:local   # Local development
npm run db:migrate:remote  # Production
```

### Development

```bash
# Frontend dev server (hot reload)
npm run dev

# Full stack with Cloudflare Pages (API + D1)
npm run build && npx wrangler pages dev ./dist --compatibility-date=2024-01-01 --local --port=8788
```

### Testing

```bash
npm run test        # Run all tests
npm run test:watch  # Watch mode
```

## Project Structure

```
makefour/
├── src/
│   ├── game/           # Connect Four engine and tests
│   ├── components/     # React components
│   ├── contexts/       # Auth, Theme contexts
│   ├── pages/          # Page components
│   └── lib/            # Utilities
├── functions/          # Cloudflare Workers API
├── documents/          # Research documentation
├── migrations/         # Database migrations
└── schema.sql          # D1 database schema
```

## Roadmap

### Phase 1: Platform (Current)
- [x] Connect Four game engine
- [x] Game state validation and win detection
- [x] User authentication
- [ ] Play vs random agent
- [ ] Game history storage
- [ ] Basic ELO tracking

### Phase 2: Oracle Integration
- [ ] Perfect play solver integration
- [ ] Move quality scoring (distance to optimal)
- [ ] Analysis mode for reviewing games

### Phase 3: Agent Zoo
- [ ] Tiny MLP agents
- [ ] CNN agents with local receptive fields
- [ ] Training pipeline with self-play
- [ ] ELO ladder for agent comparison

### Phase 4: Research Tools
- [ ] Scaling curve visualization
- [ ] Architecture comparison dashboard
- [ ] Heuristic emergence detection
- [ ] Public agent leaderboard

## License

MIT

## Contributing

This is a research project in early development. Contributions and ideas welcome!
