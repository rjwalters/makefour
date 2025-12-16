-- Migration: Add challenges table for direct player-to-player challenges
-- This enables the "Challenge Player" feature where users can challenge specific opponents

-- Challenges table
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

-- Indexes for efficient challenge lookups
CREATE INDEX IF NOT EXISTS idx_challenges_challenger ON challenges(challenger_id, status);
CREATE INDEX IF NOT EXISTS idx_challenges_target ON challenges(target_id, status);
CREATE INDEX IF NOT EXISTS idx_challenges_expires ON challenges(expires_at);
CREATE INDEX IF NOT EXISTS idx_challenges_status ON challenges(status, created_at);
