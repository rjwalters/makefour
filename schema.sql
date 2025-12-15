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
  created_at INTEGER NOT NULL,
  last_login INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  CHECK (email_verified IN (0, 1)),
  CHECK (oauth_provider IS NULL OR oauth_provider IN ('google')),
  CHECK (password_hash IS NOT NULL OR oauth_provider IS NOT NULL)
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
  -- Type of opponent: 'human' for hotseat, 'ai' for AI opponent
  opponent_type TEXT NOT NULL DEFAULT 'ai',
  -- AI difficulty level (null for human games)
  ai_difficulty TEXT,
  -- Which player the user played as (1 = red/first, 2 = yellow/second)
  player_number INTEGER NOT NULL DEFAULT 1,
  created_at INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  CHECK (outcome IN ('win', 'loss', 'draw')),
  CHECK (opponent_type IN ('human', 'ai')),
  CHECK (ai_difficulty IS NULL OR ai_difficulty IN ('beginner', 'intermediate', 'expert', 'perfect')),
  CHECK (player_number IN (1, 2))
);

-- Indexes for users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_oauth ON users(oauth_provider, oauth_id);

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
