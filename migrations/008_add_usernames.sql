-- Migration: Add username support to users table
-- Usernames are optional, unique (case-insensitive), and changeable once every 30 days

-- Add username column (NULL = use email prefix as fallback)
ALTER TABLE users ADD COLUMN username TEXT;

-- Add timestamp for tracking when username was last changed (for 30-day cooldown)
ALTER TABLE users ADD COLUMN username_changed_at INTEGER;

-- Create case-insensitive unique index for username lookups
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username_lower ON users(LOWER(username))
  WHERE username IS NOT NULL;
