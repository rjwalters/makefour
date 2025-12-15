-- Migration: Add bot vs bot game support
-- This migration adds fields to support automated bot vs bot games with spectator support

-- Add new columns to active_games for bot vs bot games
ALTER TABLE active_games ADD COLUMN is_bot_vs_bot INTEGER NOT NULL DEFAULT 0;
ALTER TABLE active_games ADD COLUMN bot1_persona_id TEXT;
ALTER TABLE active_games ADD COLUMN bot2_persona_id TEXT;
ALTER TABLE active_games ADD COLUMN move_delay_ms INTEGER;
ALTER TABLE active_games ADD COLUMN next_move_at INTEGER;

-- Add index for efficient bot vs bot game queries
CREATE INDEX IF NOT EXISTS idx_active_games_bot_vs_bot ON active_games(is_bot_vs_bot, status, next_move_at);
