-- Add timer support for chess-clock style games
-- This migration adds time control fields to active_games

-- Timer fields for tracking remaining time per player
ALTER TABLE active_games ADD COLUMN time_control_ms INTEGER;
ALTER TABLE active_games ADD COLUMN player1_time_ms INTEGER;
ALTER TABLE active_games ADD COLUMN player2_time_ms INTEGER;
ALTER TABLE active_games ADD COLUMN turn_started_at INTEGER;

-- Bot game fields for server-side ranked bot games
ALTER TABLE active_games ADD COLUMN is_bot_game INTEGER NOT NULL DEFAULT 0;
ALTER TABLE active_games ADD COLUMN bot_difficulty TEXT;

-- Index for finding timed games that may have timed out
CREATE INDEX IF NOT EXISTS idx_active_games_timed ON active_games(status, time_control_ms, turn_started_at)
  WHERE status = 'active' AND time_control_ms IS NOT NULL;

-- Index for bot games
CREATE INDEX IF NOT EXISTS idx_active_games_bot ON active_games(is_bot_game, status)
  WHERE is_bot_game = 1;
