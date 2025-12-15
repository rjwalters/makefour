-- Add bot personas infrastructure
-- This migration creates the bot_personas table for customizable AI opponents

-- Bot personas table
CREATE TABLE IF NOT EXISTS bot_personas (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  avatar_url TEXT,
  ai_engine TEXT NOT NULL DEFAULT 'minimax',
  ai_config TEXT NOT NULL DEFAULT '{}',
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
  CHECK (play_style IN ('aggressive', 'defensive', 'balanced', 'tricky', 'adaptive'))
);

-- Add bot_persona_id to active_games (nullable for backward compatibility)
ALTER TABLE active_games ADD COLUMN bot_persona_id TEXT REFERENCES bot_personas(id);

-- Add bot_persona_id to games history (nullable for backward compatibility)
ALTER TABLE games ADD COLUMN bot_persona_id TEXT REFERENCES bot_personas(id);

-- Index for finding active personas
CREATE INDEX IF NOT EXISTS idx_bot_personas_active ON bot_personas(is_active, current_elo)
  WHERE is_active = 1;

-- Index for active games by persona
CREATE INDEX IF NOT EXISTS idx_active_games_persona ON active_games(bot_persona_id)
  WHERE bot_persona_id IS NOT NULL;

-- Seed default bot personas
INSERT OR REPLACE INTO bot_personas (id, name, description, avatar_url, ai_engine, ai_config, chat_personality, play_style, base_elo, current_elo, games_played, wins, losses, draws, is_active, created_at, updated_at)
VALUES
  ('rookie', 'Rookie', 'A friendly beginner still learning the ropes. Makes lots of mistakes but never gives up!', NULL, 'minimax', '{"searchDepth":2,"errorRate":0.35,"timeMultiplier":0.1}', '{"style":"enthusiastic","greeting":"Hi! I''m still learning, so go easy on me!","onWin":"Wow, I won! I''m getting better!","onLose":"Good game! I''ll get you next time!","onGoodMove":"Nice move! I need to pay more attention.","onBadMove":"Oops, that wasn''t my best move...","tauntFrequency":0.1}', 'balanced', 700, 700, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000),

  ('rusty', 'Rusty', 'An old-timer getting back into the game. Solid fundamentals but occasionally rusty.', NULL, 'minimax', '{"searchDepth":3,"errorRate":0.25,"timeMultiplier":0.2}', '{"style":"nostalgic","greeting":"Ah, this takes me back. Let''s see if I still got it.","onWin":"Still got the old magic!","onLose":"Well played. The new generation is sharp.","onGoodMove":"That reminds me of a game I played years ago...","onBadMove":"Hmm, I used to be sharper than that.","tauntFrequency":0.15}', 'defensive', 900, 900, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000),

  ('nova', 'Nova', 'A promising player with flashes of brilliance. Occasionally makes mistakes under pressure.', NULL, 'minimax', '{"searchDepth":4,"errorRate":0.15,"timeMultiplier":0.3}', '{"style":"confident","greeting":"Ready to shine! Let''s have a great game.","onWin":"That''s what I''m talking about!","onLose":"Impressive! I''ll study that game.","onGoodMove":"Okay, I see you!","onBadMove":"Wait, that wasn''t the plan...","tauntFrequency":0.2}', 'aggressive', 1100, 1100, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000),

  ('scholar', 'Scholar', 'A methodical player who has studied every opening. Rarely makes mistakes.', NULL, 'minimax', '{"searchDepth":6,"errorRate":0.08,"timeMultiplier":0.5}', '{"style":"analytical","greeting":"Interesting. Let''s explore the position together.","onWin":"A well-calculated victory.","onLose":"Fascinating. I must reconsider my evaluation.","onGoodMove":"An interesting choice. Let me think...","onBadMove":"According to my analysis, that was suboptimal.","tauntFrequency":0.1}', 'balanced', 1350, 1350, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000),

  ('viper', 'Viper', 'A cunning strategist who sets traps and thrives on opponent mistakes.', NULL, 'minimax', '{"searchDepth":5,"errorRate":0.1,"timeMultiplier":0.4}', '{"style":"cunning","greeting":"Let''s play a little game...","onWin":"You walked right into my trap.","onLose":"Well played. You saw through my schemes.","onGoodMove":"Careful now... one wrong step...","onBadMove":"Hmm, you might regret that.","tauntFrequency":0.35}', 'tricky', 1250, 1250, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000),

  ('titan', 'Titan', 'A powerful player who dominates the center and crushes opposition.', NULL, 'minimax', '{"searchDepth":7,"errorRate":0.04,"timeMultiplier":0.6}', '{"style":"intimidating","greeting":"Prepare yourself.","onWin":"As expected.","onLose":"Impressive. You have earned my respect.","onGoodMove":"Not bad.","onBadMove":"Your position crumbles.","tauntFrequency":0.25}', 'aggressive', 1550, 1550, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000),

  ('sentinel', 'Sentinel', 'An unshakeable defender who never makes mistakes. Nearly impossible to beat.', NULL, 'minimax', '{"searchDepth":10,"errorRate":0.01,"timeMultiplier":0.8}', '{"style":"stoic","greeting":"I am ready.","onWin":"The fortress holds.","onLose":"You have breached my defenses. Well done.","onGoodMove":"A worthy attempt.","onBadMove":"Your strategy falters.","tauntFrequency":0.15}', 'defensive', 1800, 1800, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000),

  ('oracle', 'Oracle', 'Plays with perfect precision. Sees every move before you make it.', NULL, 'minimax', '{"searchDepth":42,"errorRate":0,"timeMultiplier":0.95}', '{"style":"mysterious","greeting":"I have foreseen this game.","onWin":"It was written.","onLose":"Impossible... the visions were wrong.","onGoodMove":"As I predicted.","onBadMove":"Your path leads to defeat.","tauntFrequency":0.2}', 'adaptive', 2200, 2200, 0, 0, 0, 0, 1, strftime('%s','now') * 1000, strftime('%s','now') * 1000);
