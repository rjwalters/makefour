-- Add player vs bot stats tracking
-- Tracks per-player records against each bot for progression and achievements

-- Player vs bot stats table
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

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_player_bot_stats_user ON player_bot_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_player_bot_stats_bot ON player_bot_stats(bot_persona_id);
CREATE INDEX IF NOT EXISTS idx_player_bot_stats_last_played ON player_bot_stats(last_played_at DESC);

-- Index for looking up games by bot persona
CREATE INDEX IF NOT EXISTS idx_games_bot_persona ON games(bot_persona_id)
  WHERE bot_persona_id IS NOT NULL;
