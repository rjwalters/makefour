-- Add bot user infrastructure to users table
-- This migration adds is_bot and bot_persona_id columns and creates user accounts for each bot persona

-- Add is_bot column to users table
ALTER TABLE users ADD COLUMN is_bot INTEGER NOT NULL DEFAULT 0;

-- Add bot_persona_id column to users table
ALTER TABLE users ADD COLUMN bot_persona_id TEXT REFERENCES bot_personas(id);

-- Create indexes for bot-related queries
CREATE INDEX IF NOT EXISTS idx_users_is_bot ON users(is_bot);
CREATE INDEX IF NOT EXISTS idx_users_bot_persona ON users(bot_persona_id);

-- Create bot user accounts for each existing persona
-- Bot user ID format: bot_<persona_id>
-- Email format: <persona_id>@bot.makefour.game
INSERT INTO users (id, email, email_verified, rating, games_played, wins, losses, draws, is_bot, bot_persona_id, created_at, last_login, updated_at)
SELECT
  'bot_' || id,
  id || '@bot.makefour.game',
  1,  -- email_verified (bots are always "verified")
  current_elo,
  games_played,
  wins,
  losses,
  draws,
  1,  -- is_bot
  id,
  created_at,
  created_at,
  updated_at
FROM bot_personas
WHERE is_active = 1
ON CONFLICT (id) DO UPDATE SET
  rating = excluded.rating,
  games_played = excluded.games_played,
  wins = excluded.wins,
  losses = excluded.losses,
  draws = excluded.draws,
  updated_at = excluded.updated_at;
