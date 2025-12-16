-- Migration: Add opponent_id to games table for tracking head-to-head matchups
-- This enables bot vs bot matchup statistics

-- Add opponent_id column to track who the opponent was in each game
ALTER TABLE games ADD COLUMN opponent_id TEXT;

-- Index for efficient matchup queries (find all games between two bots)
CREATE INDEX IF NOT EXISTS idx_games_opponent_id ON games(opponent_id);

-- Composite index for bot matchup queries
CREATE INDEX IF NOT EXISTS idx_games_user_opponent ON games(user_id, opponent_id);
