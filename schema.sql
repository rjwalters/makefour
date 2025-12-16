-- D1 Database Schema for MakeFour

-- Users table with email/password authentication
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT UNIQUE NOT NULL,
  email_verified INTEGER NOT NULL DEFAULT 0,
  password_hash TEXT,
  oauth_provider TEXT,
  oauth_id TEXT,
  encrypted_dek TEXT,
  -- ELO rating fields
  rating INTEGER NOT NULL DEFAULT 1200,
  games_played INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  -- User preferences (JSON)
  preferences TEXT DEFAULT '{}',
  -- Bot user fields
  is_bot INTEGER NOT NULL DEFAULT 0,
  bot_persona_id TEXT,
  created_at INTEGER NOT NULL,
  last_login INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  CHECK (email_verified IN (0, 1)),
  CHECK (oauth_provider IS NULL OR oauth_provider IN ('google')),
  CHECK (is_bot IN (0, 1)),
  CHECK (password_hash IS NOT NULL OR oauth_provider IS NOT NULL OR is_bot = 1),
  FOREIGN KEY (bot_persona_id) REFERENCES bot_personas(id)
);

-- Session tokens for persistent login
CREATE TABLE IF NOT EXISTS session_tokens (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Email verification tokens
CREATE TABLE IF NOT EXISTS email_verification_tokens (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  used INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CHECK (used IN (0, 1))
);

-- Password reset tokens
CREATE TABLE IF NOT EXISTS password_reset_tokens (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  expires_at INTEGER NOT NULL,
  used INTEGER NOT NULL DEFAULT 0,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CHECK (used IN (0, 1))
);

-- Games table - stores completed games for logged-in users
CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  -- Outcome from the perspective of the user
  outcome TEXT NOT NULL,
  -- JSON array of column indices (0-6) representing each move
  moves TEXT NOT NULL,
  -- Total number of moves in the game
  move_count INTEGER NOT NULL,
  -- ELO rating change from this game (positive for gains, negative for losses)
  rating_change INTEGER DEFAULT 0,
  -- Type of opponent: 'human' for hotseat, 'ai' for AI opponent
  opponent_type TEXT NOT NULL DEFAULT 'ai',
  -- Opponent user ID (for bot vs bot matchup tracking)
  opponent_id TEXT,
  -- AI difficulty level (null for human games)
  ai_difficulty TEXT,
  -- Bot persona ID for ranked bot games (null for training/human games)
  bot_persona_id TEXT,
  -- Which player the user played as (1 = red/first, 2 = yellow/second)
  player_number INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (bot_persona_id) REFERENCES bot_personas(id),
  CHECK (outcome IN ('win', 'loss', 'draw')),
  CHECK (opponent_type IN ('human', 'ai')),
  CHECK (ai_difficulty IS NULL OR ai_difficulty IN ('beginner', 'intermediate', 'expert', 'perfect')),
  CHECK (player_number IN (1, 2))
);

-- Indexes for users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_oauth ON users(oauth_provider, oauth_id);
CREATE INDEX IF NOT EXISTS idx_users_is_bot ON users(is_bot);
CREATE INDEX IF NOT EXISTS idx_users_bot_persona ON users(bot_persona_id);

-- Indexes for session tokens
CREATE INDEX IF NOT EXISTS idx_session_tokens_user ON session_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_session_tokens_expires ON session_tokens(expires_at);

-- Indexes for verification tokens
CREATE INDEX IF NOT EXISTS idx_verification_tokens_user ON email_verification_tokens(user_id);
CREATE INDEX IF NOT EXISTS idx_reset_tokens_user ON password_reset_tokens(user_id);

-- Indexes for games
CREATE INDEX IF NOT EXISTS idx_games_user_id ON games(user_id);
CREATE INDEX IF NOT EXISTS idx_games_created_at ON games(created_at);
CREATE INDEX IF NOT EXISTS idx_games_outcome ON games(outcome);
CREATE INDEX IF NOT EXISTS idx_games_opponent_type ON games(opponent_type);
CREATE INDEX IF NOT EXISTS idx_games_ai_difficulty ON games(ai_difficulty);
CREATE INDEX IF NOT EXISTS idx_games_bot_persona ON games(bot_persona_id);
-- Index for bot matchup queries
CREATE INDEX IF NOT EXISTS idx_games_opponent_id ON games(opponent_id);
CREATE INDEX IF NOT EXISTS idx_games_user_opponent ON games(user_id, opponent_id);

-- Rating history table - tracks ELO changes after each game
CREATE TABLE IF NOT EXISTS rating_history (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  game_id TEXT NOT NULL,
  rating_before INTEGER NOT NULL,
  rating_after INTEGER NOT NULL,
  rating_change INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (game_id) REFERENCES games(id) ON DELETE CASCADE
);

-- Indexes for rating history
CREATE INDEX IF NOT EXISTS idx_rating_history_user_id ON rating_history(user_id);
CREATE INDEX IF NOT EXISTS idx_rating_history_created_at ON rating_history(created_at);

-- Index for leaderboard queries (descending by rating)
CREATE INDEX IF NOT EXISTS idx_users_rating ON users(rating DESC);

-- Matchmaking queue - players waiting for an opponent
CREATE TABLE IF NOT EXISTS matchmaking_queue (
  id TEXT PRIMARY KEY,
  user_id TEXT UNIQUE NOT NULL,
  rating INTEGER NOT NULL,
  mode TEXT NOT NULL DEFAULT 'ranked',
  -- Rating tolerance expands over time
  initial_tolerance INTEGER NOT NULL DEFAULT 100,
  -- Whether the resulting game should be spectatable
  spectatable INTEGER NOT NULL DEFAULT 1,
  -- When the user joined the queue
  joined_at INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CHECK (mode IN ('ranked', 'casual')),
  CHECK (spectatable IN (0, 1))
);

-- Indexes for matchmaking queue
CREATE INDEX IF NOT EXISTS idx_matchmaking_queue_rating ON matchmaking_queue(rating);
CREATE INDEX IF NOT EXISTS idx_matchmaking_queue_mode ON matchmaking_queue(mode);
CREATE INDEX IF NOT EXISTS idx_matchmaking_queue_joined_at ON matchmaking_queue(joined_at);

-- Active games - games currently in progress
CREATE TABLE IF NOT EXISTS active_games (
  id TEXT PRIMARY KEY,
  player1_id TEXT NOT NULL,
  player2_id TEXT NOT NULL,
  -- JSON array of column indices (0-6) representing each move
  moves TEXT NOT NULL DEFAULT '[]',
  -- Current turn: 1 or 2
  current_turn INTEGER NOT NULL DEFAULT 1,
  -- Game status
  status TEXT NOT NULL DEFAULT 'active',
  -- Game mode (affects ELO)
  mode TEXT NOT NULL DEFAULT 'ranked',
  -- Winner (null if ongoing, 1, 2, or 'draw')
  winner TEXT,
  -- Rating snapshots at game start (for ELO calculation)
  player1_rating INTEGER NOT NULL,
  player2_rating INTEGER NOT NULL,
  -- Whether spectators can watch this game
  spectatable INTEGER NOT NULL DEFAULT 1,
  -- Count of current spectators (for display purposes)
  spectator_count INTEGER NOT NULL DEFAULT 0,
  -- Last activity timestamp (for abandonment detection)
  last_move_at INTEGER NOT NULL,
  -- Timer fields (NULL = untimed game, 300000 = 5 minutes)
  time_control_ms INTEGER,
  player1_time_ms INTEGER,
  player2_time_ms INTEGER,
  turn_started_at INTEGER,
  -- Bot game fields (for server-side ranked bot games)
  is_bot_game INTEGER NOT NULL DEFAULT 0,
  bot_difficulty TEXT,
  -- Bot vs Bot game fields
  is_bot_vs_bot INTEGER NOT NULL DEFAULT 0,
  bot1_persona_id TEXT,
  bot2_persona_id TEXT,
  -- Pacing for bot vs bot games (ms delay between moves)
  move_delay_ms INTEGER,
  -- Last move timestamp for spectator display
  next_move_at INTEGER,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY (player1_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (player2_id) REFERENCES users(id) ON DELETE CASCADE,
  CHECK (current_turn IN (1, 2)),
  CHECK (status IN ('active', 'completed', 'abandoned')),
  CHECK (mode IN ('ranked', 'casual')),
  CHECK (winner IS NULL OR winner IN ('1', '2', 'draw')),
  CHECK (spectatable IN (0, 1)),
  CHECK (is_bot_game IN (0, 1)),
  CHECK (bot_difficulty IS NULL OR bot_difficulty IN ('beginner', 'intermediate', 'expert', 'perfect')),
  CHECK (is_bot_vs_bot IN (0, 1))
);

-- Indexes for active games
CREATE INDEX IF NOT EXISTS idx_active_games_player1 ON active_games(player1_id);
CREATE INDEX IF NOT EXISTS idx_active_games_player2 ON active_games(player2_id);
CREATE INDEX IF NOT EXISTS idx_active_games_status ON active_games(status);
CREATE INDEX IF NOT EXISTS idx_active_games_updated_at ON active_games(updated_at);
-- Index for spectatable games (for live game browsing)
CREATE INDEX IF NOT EXISTS idx_active_games_spectatable ON active_games(spectatable, status);
-- Index for finding timed games (timeout detection)
CREATE INDEX IF NOT EXISTS idx_active_games_timed ON active_games(status, time_control_ms, turn_started_at);
-- Index for bot games
CREATE INDEX IF NOT EXISTS idx_active_games_bot ON active_games(is_bot_game, status);
-- Index for bot vs bot games (for spectating and game orchestration)
CREATE INDEX IF NOT EXISTS idx_active_games_bot_vs_bot ON active_games(is_bot_vs_bot, status, next_move_at);

-- Game messages - chat messages during active games
CREATE TABLE IF NOT EXISTS game_messages (
  id TEXT PRIMARY KEY,
  game_id TEXT NOT NULL,
  sender_id TEXT NOT NULL,  -- user_id or 'bot'
  sender_type TEXT NOT NULL,  -- 'human' or 'bot'
  content TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (game_id) REFERENCES active_games(id) ON DELETE CASCADE,
  CHECK (sender_type IN ('human', 'bot'))
);

-- Indexes for game messages
CREATE INDEX IF NOT EXISTS idx_game_messages_game_id ON game_messages(game_id);
CREATE INDEX IF NOT EXISTS idx_game_messages_created_at ON game_messages(created_at);

-- Bot personas table - customizable AI opponents with unique personalities
CREATE TABLE IF NOT EXISTS bot_personas (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  avatar_url TEXT,
  ai_engine TEXT NOT NULL DEFAULT 'minimax',
  -- JSON object with engine-specific parameters (searchDepth, errorRate, etc.)
  ai_config TEXT NOT NULL DEFAULT '{}',
  -- JSON object with chat personality (style, phrases, reactions)
  chat_personality TEXT NOT NULL DEFAULT '{}',
  play_style TEXT NOT NULL DEFAULT 'balanced',
  base_elo INTEGER NOT NULL DEFAULT 1200,
  current_elo INTEGER NOT NULL DEFAULT 1200,
  games_played INTEGER NOT NULL DEFAULT 0,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  CHECK (play_style IN ('aggressive', 'defensive', 'balanced', 'tricky', 'adaptive')),
  CHECK (is_active IN (0, 1))
);

-- Index for finding active personas
CREATE INDEX IF NOT EXISTS idx_bot_personas_active ON bot_personas(is_active, current_elo);

-- Player vs bot stats - tracks per-player records against each bot
CREATE TABLE IF NOT EXISTS player_bot_stats (
  user_id TEXT NOT NULL,
  bot_persona_id TEXT NOT NULL,
  wins INTEGER NOT NULL DEFAULT 0,
  losses INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  -- Current streak: positive = win streak, negative = loss streak
  current_streak INTEGER NOT NULL DEFAULT 0,
  best_win_streak INTEGER NOT NULL DEFAULT 0,
  -- Timestamp of first win against this bot (milestone)
  first_win_at INTEGER,
  last_played_at INTEGER NOT NULL,
  PRIMARY KEY (user_id, bot_persona_id),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (bot_persona_id) REFERENCES bot_personas(id) ON DELETE CASCADE
);

-- Indexes for player bot stats
CREATE INDEX IF NOT EXISTS idx_player_bot_stats_user ON player_bot_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_player_bot_stats_bot ON player_bot_stats(bot_persona_id);
CREATE INDEX IF NOT EXISTS idx_player_bot_stats_last_played ON player_bot_stats(last_played_at DESC);

-- Challenges table - direct player-to-player challenges
CREATE TABLE IF NOT EXISTS challenges (
  id TEXT PRIMARY KEY,
  -- User who initiated the challenge
  challenger_id TEXT NOT NULL,
  challenger_username TEXT NOT NULL,
  challenger_rating INTEGER NOT NULL,
  -- Target user (can be NULL if username doesn't exist yet)
  target_id TEXT,
  target_username TEXT NOT NULL,
  target_rating INTEGER,
  -- Challenge status
  status TEXT NOT NULL DEFAULT 'pending',
  -- Timestamps
  created_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  -- Game ID if challenge was accepted
  game_id TEXT,
  FOREIGN KEY (challenger_id) REFERENCES users(id) ON DELETE CASCADE,
  FOREIGN KEY (target_id) REFERENCES users(id) ON DELETE SET NULL,
  FOREIGN KEY (game_id) REFERENCES active_games(id) ON DELETE SET NULL,
  CHECK (status IN ('pending', 'accepted', 'cancelled', 'declined', 'expired'))
);

-- Indexes for challenges
CREATE INDEX IF NOT EXISTS idx_challenges_challenger ON challenges(challenger_id, status);
CREATE INDEX IF NOT EXISTS idx_challenges_target ON challenges(target_id, status);
CREATE INDEX IF NOT EXISTS idx_challenges_expires ON challenges(expires_at);
CREATE INDEX IF NOT EXISTS idx_challenges_status ON challenges(status, created_at);
