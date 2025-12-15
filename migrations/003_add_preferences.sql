-- Add user preferences for cross-device sync
-- This migration adds a preferences column to store user settings as JSON

-- Preferences column stores JSON with the following schema:
-- {
--   "soundEnabled": boolean (default: true),
--   "soundVolume": number 0-100 (default: 50),
--   "defaultGameMode": "ai" | "hotseat" | "online" (default: "ai"),
--   "defaultDifficulty": "beginner" | "intermediate" | "expert" | "perfect" (default: "intermediate"),
--   "defaultPlayerColor": 1 | 2 (default: 1),
--   "defaultMatchmakingMode": "ranked" | "casual" (default: "ranked"),
--   "allowSpectators": boolean (default: true),
--   "theme": "light" | "dark" | "system" (default: "system")
-- }

ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT '{}';
