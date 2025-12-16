import type { InferSelectModel, InferInsertModel } from 'drizzle-orm'
import {
  users,
  sessionTokens,
  emailVerificationTokens,
  passwordResetTokens,
  games,
  activeGames,
  gameMessages,
  matchmakingQueue,
  challenges,
  ratingHistory,
  botPersonas,
  playerBotStats,
} from './schema'

// =============================================================================
// User Types
// =============================================================================

export type User = InferSelectModel<typeof users>
export type NewUser = InferInsertModel<typeof users>

/** User type with sensitive fields omitted (for API responses) */
export type PublicUser = Omit<User, 'passwordHash' | 'oauthId' | 'encryptedDek'>

// =============================================================================
// Session & Auth Token Types
// =============================================================================

export type SessionToken = InferSelectModel<typeof sessionTokens>
export type NewSessionToken = InferInsertModel<typeof sessionTokens>

export type EmailVerificationToken = InferSelectModel<typeof emailVerificationTokens>
export type NewEmailVerificationToken = InferInsertModel<typeof emailVerificationTokens>

export type PasswordResetToken = InferSelectModel<typeof passwordResetTokens>
export type NewPasswordResetToken = InferInsertModel<typeof passwordResetTokens>

// =============================================================================
// Game Types
// =============================================================================

export type Game = InferSelectModel<typeof games>
export type NewGame = InferInsertModel<typeof games>

export type ActiveGame = InferSelectModel<typeof activeGames>
export type NewActiveGame = InferInsertModel<typeof activeGames>

export type GameMessage = InferSelectModel<typeof gameMessages>
export type NewGameMessage = InferInsertModel<typeof gameMessages>

// =============================================================================
// Matchmaking & Challenge Types
// =============================================================================

export type MatchmakingQueueEntry = InferSelectModel<typeof matchmakingQueue>
export type NewMatchmakingQueueEntry = InferInsertModel<typeof matchmakingQueue>

export type Challenge = InferSelectModel<typeof challenges>
export type NewChallenge = InferInsertModel<typeof challenges>

// =============================================================================
// Rating Types
// =============================================================================

export type RatingHistoryEntry = InferSelectModel<typeof ratingHistory>
export type NewRatingHistoryEntry = InferInsertModel<typeof ratingHistory>

// =============================================================================
// Bot Types
// =============================================================================

export type BotPersona = InferSelectModel<typeof botPersonas>
export type NewBotPersona = InferInsertModel<typeof botPersonas>

export type PlayerBotStats = InferSelectModel<typeof playerBotStats>
export type NewPlayerBotStats = InferInsertModel<typeof playerBotStats>

// =============================================================================
// Enum Types (for type safety)
// =============================================================================

export type GameOutcome = 'win' | 'loss' | 'draw'
export type OpponentType = 'human' | 'ai'
export type AIDifficulty = 'beginner' | 'intermediate' | 'expert' | 'perfect'
export type GameStatus = 'active' | 'completed' | 'abandoned'
export type GameMode = 'ranked' | 'casual'
export type GameWinner = '1' | '2' | 'draw' | null
export type PlayerNumber = 1 | 2
export type SenderType = 'human' | 'bot'
export type PlayStyle = 'aggressive' | 'defensive' | 'balanced' | 'tricky' | 'adaptive'
export type ChallengeStatus = 'pending' | 'accepted' | 'cancelled' | 'declined' | 'expired'
export type OAuthProvider = 'google'

// =============================================================================
// Re-export JSON schema types
// =============================================================================

export type {
  UserPreferences,
  Moves,
  AIConfig,
  ChatPersonality,
} from './json-schemas'

export {
  parseUserPreferences,
  parseMoves,
  parseAIConfig,
  parseChatPersonality,
  serializeUserPreferences,
  serializeMoves,
  serializeAIConfig,
  serializeChatPersonality,
  DEFAULT_USER_PREFERENCES,
  userPreferencesSchema,
  movesSchema,
  aiConfigSchema,
  chatPersonalitySchema,
} from './json-schemas'
