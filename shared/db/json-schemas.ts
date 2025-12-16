import { z } from 'zod'

// =============================================================================
// User Preferences Schema
// =============================================================================

export const userPreferencesSchema = z.object({
  soundEnabled: z.boolean().default(true),
  soundVolume: z.number().min(0).max(100).default(50),
  defaultGameMode: z.enum(['ai', 'hotseat', 'online']).default('ai'),
  defaultDifficulty: z.enum(['beginner', 'intermediate', 'expert', 'perfect']).default('intermediate'),
  defaultPlayerColor: z.union([z.literal(1), z.literal(2)]).default(1),
  defaultMatchmakingMode: z.enum(['ranked', 'casual']).default('ranked'),
  allowSpectators: z.boolean().default(true),
  theme: z.enum(['light', 'dark', 'system']).default('system'),
})

export type UserPreferences = z.infer<typeof userPreferencesSchema>

export const DEFAULT_USER_PREFERENCES: UserPreferences = {
  soundEnabled: true,
  soundVolume: 50,
  defaultGameMode: 'ai',
  defaultDifficulty: 'intermediate',
  defaultPlayerColor: 1,
  defaultMatchmakingMode: 'ranked',
  allowSpectators: true,
  theme: 'system',
}

// =============================================================================
// Game Moves Schema
// =============================================================================

export const movesSchema = z.array(z.number().int().min(0).max(6))

export type Moves = z.infer<typeof movesSchema>

// =============================================================================
// AI Config Schema
// =============================================================================

export const aiConfigSchema = z.object({
  searchDepth: z.number().int().min(1).max(42).optional(),
  errorRate: z.number().min(0).max(1).optional(),
  timeMultiplier: z.number().optional(),
  evalWeights: z.object({
    ownThreats: z.number().optional(),
    opponentThreats: z.number().optional(),
    centerControl: z.number().optional(),
    doubleThreats: z.number().optional(),
  }).optional(),
  useTranspositionTable: z.boolean().optional(),
})

export type AIConfig = z.infer<typeof aiConfigSchema>

// =============================================================================
// Chat Personality Schema
// =============================================================================

export const chatPersonalitySchema = z.object({
  name: z.string().optional(),
  systemPrompt: z.string().optional(),
  reactions: z.object({
    gameStart: z.array(z.string()).optional(),
    playerGoodMove: z.array(z.string()).optional(),
    playerBlunder: z.array(z.string()).optional(),
    botWinning: z.array(z.string()).optional(),
    botLosing: z.array(z.string()).optional(),
    gameWon: z.array(z.string()).optional(),
    gameLost: z.array(z.string()).optional(),
    draw: z.array(z.string()).optional(),
  }).optional(),
  chattiness: z.number().min(0).max(1).optional(),
  useEmoji: z.boolean().optional(),
  maxLength: z.number().int().positive().optional(),
  temperature: z.number().min(0).max(1).optional(),
  // Legacy format support
  style: z.string().optional(),
  greeting: z.string().optional(),
  onWin: z.string().optional(),
  onLose: z.string().optional(),
  onGoodMove: z.string().optional(),
  onBadMove: z.string().optional(),
  tauntFrequency: z.number().min(0).max(1).optional(),
})

export type ChatPersonality = z.infer<typeof chatPersonalitySchema>

// =============================================================================
// Parsing Helpers
// =============================================================================

export function parseUserPreferences(json: string | null | undefined): UserPreferences {
  if (!json) return DEFAULT_USER_PREFERENCES
  try {
    const parsed = JSON.parse(json)
    return { ...DEFAULT_USER_PREFERENCES, ...userPreferencesSchema.partial().parse(parsed) }
  } catch {
    return DEFAULT_USER_PREFERENCES
  }
}

export function parseMoves(json: string | null | undefined): Moves {
  if (!json) return []
  try {
    return movesSchema.parse(JSON.parse(json))
  } catch {
    return []
  }
}

export function parseAIConfig(json: string | null | undefined): AIConfig {
  if (!json) return {}
  try {
    return aiConfigSchema.parse(JSON.parse(json))
  } catch {
    return {}
  }
}

export function parseChatPersonality(json: string | null | undefined): ChatPersonality {
  if (!json) return {}
  try {
    return chatPersonalitySchema.parse(JSON.parse(json))
  } catch {
    return {}
  }
}

// =============================================================================
// Serialization Helpers
// =============================================================================

export function serializeUserPreferences(prefs: Partial<UserPreferences>): string {
  return JSON.stringify({ ...DEFAULT_USER_PREFERENCES, ...prefs })
}

export function serializeMoves(moves: Moves): string {
  return JSON.stringify(moves)
}

export function serializeAIConfig(config: AIConfig): string {
  return JSON.stringify(config)
}

export function serializeChatPersonality(personality: ChatPersonality): string {
  return JSON.stringify(personality)
}
