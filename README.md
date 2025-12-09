# MakeFour

A four-in-a-row strategy site with AI coaching experiments, built on Cloudflare Pages.

## Vision

MakeFour is a testbed for studying intelligence in a solved game domain. Four-in-a-row is:

- **Fully solved** — perfect play oracle exists for ground truth
- **Structured** — gravity, discrete columns, local interactions
- **Measurable** — ELO provides continuous performance metrics
- **Tractable** — small enough to train many agent variants

Future research goals include:
- Minimum viable neural networks for specific ELO thresholds
- Scaling curves across architectural families (MLP, CNN, etc.)
- Identifying cognitive phase transitions where specific heuristics emerge

## Current Features

- **User Authentication**: Email/password login with encrypted sessions
- **Play MakeFour**: Full 7x6 four-in-a-row game (hotseat mode)
- **Game History**: Saved games with move-by-move replay
- **AI Coach (Stub)**: Placeholder for future game analysis

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS + shadcn/ui components
- **Backend**: Cloudflare Pages Functions (Workers)
- **Database**: Cloudflare D1 (SQLite)
- **Auth**: Email/password with bcrypt, 30-day sessions
- **Encryption**: AES-GCM two-tier model (DEK/KEK)

## Project Structure

```
makefour/
├── src/
│   ├── pages/              # React page components
│   │   ├── LoginPage.tsx   # Authentication
│   │   ├── DashboardPage.tsx
│   │   ├── PlayPage.tsx    # Game board
│   │   ├── GamesPage.tsx   # History list
│   │   └── ReplayPage.tsx  # Move replay
│   ├── components/
│   │   ├── GameBoard.tsx   # 7x6 grid UI
│   │   └── ui/             # shadcn components
│   ├── game/
│   │   ├── makefour.ts     # Pure game engine
│   │   └── makefour.test.ts
│   ├── ai/
│   │   └── coach.ts        # AI stubs
│   ├── contexts/           # Auth, Theme
│   ├── hooks/
│   └── lib/                # Crypto, utils
├── functions/              # Cloudflare Workers
│   ├── api/
│   │   ├── auth/           # Login, register, logout
│   │   ├── games.ts        # Game CRUD
│   │   └── games/[id].ts   # Single game
│   └── lib/
├── migrations/
├── schema.sql
└── wrangler.toml
```

## Local Development

### Prerequisites

- Node.js 18+
- npm

### Setup

```bash
# Install dependencies
npm install

# Create local D1 database
npm run db:migrate:local

# Start dev servers (run in separate terminals)
npm run dev           # Vite frontend
npm run pages:dev     # Wrangler API + DB
```

Open http://localhost:5173

### Running Tests

```bash
npm test              # Run once
npm run test:watch    # Watch mode
```

## Deployment

### First-time Setup

1. Login to Cloudflare:
   ```bash
   npx wrangler login
   ```

2. Create D1 database:
   ```bash
   npm run db:create
   ```

3. Update `wrangler.toml` with your database ID:
   ```toml
   [[d1_databases]]
   binding = "DB"
   database_name = "makefour-db"
   database_id = "your-database-id"
   ```

4. Apply schema:
   ```bash
   npm run db:migrate:remote
   ```

### Deploy

```bash
npm run deploy
```

## Architecture

### Game Engine (`src/game/makefour.ts`)

Pure TypeScript, no dependencies:

```typescript
createGameState()           // New game
makeMove(state, column)     // Apply move → new state
checkWinner(board)          // Win/draw/ongoing
replayMoves(moves)          // Reconstruct from history
getStateAtMove(moves, idx)  // State at any point
```

### AI Coach (`src/ai/coach.ts`)

Stub implementation ready for future engines:

```typescript
analyzePosition(position)  // Evaluation + best move
suggestMove(position)      // Quick recommendation
rankMoves(position)        // Score all moves
```

Currently uses center-preference heuristics. Future options:
- Minimax with alpha-beta pruning
- Monte Carlo Tree Search
- Neural network evaluation
- Perfect play database lookup

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Create account |
| `/api/auth/login` | POST | Log in |
| `/api/auth/logout` | POST | Log out |
| `/api/auth/me` | GET | Current user |
| `/api/games` | GET | List games |
| `/api/games` | POST | Save game |
| `/api/games/:id` | GET | Single game |

## Roadmap

### Phase 1: Platform (Current)
- [x] Game engine with tests
- [x] User authentication
- [x] Play in browser
- [x] Game history & replay
- [x] AI coach stubs

### Phase 2: AI Integration
- [ ] Perfect play solver
- [ ] Move quality scoring
- [ ] Real-time analysis

### Phase 3: Multiplayer
- [ ] ELO rating system
- [ ] Online matchmaking
- [ ] Spectating

### Phase 4: Research
- [ ] Neural network agents
- [ ] Training pipeline
- [ ] Scaling experiments

## License

MIT
