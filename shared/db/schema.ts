import { sqliteTable, text, integer, index, uniqueIndex, primaryKey } from 'drizzle-orm/sqlite-core'
import { relations, sql } from 'drizzle-orm'

// =============================================================================
// Users Table
// =============================================================================

export const users = sqliteTable('users', {
  id: text('id').primaryKey(),
  email: text('email').unique().notNull(),
  emailVerified: integer('email_verified').notNull().default(0),
  passwordHash: text('password_hash'),
  oauthProvider: text('oauth_provider'),
  oauthId: text('oauth_id'),
  encryptedDek: text('encrypted_dek'),
  // ELO rating fields
  rating: integer('rating').notNull().default(1200),
  gamesPlayed: integer('games_played').notNull().default(0),
  wins: integer('wins').notNull().default(0),
  losses: integer('losses').notNull().default(0),
  draws: integer('draws').notNull().default(0),
  // User preferences (JSON)
  preferences: text('preferences').$type<string>().default('{}'),
  // Bot user fields
  isBot: integer('is_bot').notNull().default(0),
  botPersonaId: text('bot_persona_id').references(() => botPersonas.id),
  // Username fields
  username: text('username'),
  usernameChangedAt: integer('username_changed_at'),
  // Timestamps
  createdAt: integer('created_at').notNull(),
  lastLogin: integer('last_login').notNull(),
  updatedAt: integer('updated_at').notNull(),
}, (table) => [
  index('idx_users_email').on(table.email),
  index('idx_users_oauth').on(table.oauthProvider, table.oauthId),
  index('idx_users_is_bot').on(table.isBot),
  index('idx_users_bot_persona').on(table.botPersonaId),
  uniqueIndex('idx_users_username_lower').on(sql`LOWER(${table.username})`),
  index('idx_users_rating').on(table.rating),
])

export const usersRelations = relations(users, ({ one, many }) => ({
  botPersona: one(botPersonas, {
    fields: [users.botPersonaId],
    references: [botPersonas.id],
  }),
  sessions: many(sessionTokens),
  games: many(games),
  ratingHistory: many(ratingHistory),
  playerBotStats: many(playerBotStats),
  challengesSent: many(challenges, { relationName: 'challenger' }),
  challengesReceived: many(challenges, { relationName: 'target' }),
}))

// =============================================================================
// Session Tokens Table
// =============================================================================

export const sessionTokens = sqliteTable('session_tokens', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: integer('expires_at').notNull(),
  createdAt: integer('created_at').notNull(),
}, (table) => [
  index('idx_session_tokens_user').on(table.userId),
  index('idx_session_tokens_expires').on(table.expiresAt),
])

export const sessionTokensRelations = relations(sessionTokens, ({ one }) => ({
  user: one(users, {
    fields: [sessionTokens.userId],
    references: [users.id],
  }),
}))

// =============================================================================
// Email Verification Tokens Table
// =============================================================================

export const emailVerificationTokens = sqliteTable('email_verification_tokens', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: integer('expires_at').notNull(),
  used: integer('used').notNull().default(0),
  createdAt: integer('created_at').notNull(),
}, (table) => [
  index('idx_verification_tokens_user').on(table.userId),
])

export const emailVerificationTokensRelations = relations(emailVerificationTokens, ({ one }) => ({
  user: one(users, {
    fields: [emailVerificationTokens.userId],
    references: [users.id],
  }),
}))

// =============================================================================
// Password Reset Tokens Table
// =============================================================================

export const passwordResetTokens = sqliteTable('password_reset_tokens', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: integer('expires_at').notNull(),
  used: integer('used').notNull().default(0),
  createdAt: integer('created_at').notNull(),
}, (table) => [
  index('idx_reset_tokens_user').on(table.userId),
])

export const passwordResetTokensRelations = relations(passwordResetTokens, ({ one }) => ({
  user: one(users, {
    fields: [passwordResetTokens.userId],
    references: [users.id],
  }),
}))

// =============================================================================
// Games Table (Completed Games)
// =============================================================================

export const games = sqliteTable('games', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  // Outcome from the perspective of the user
  outcome: text('outcome').notNull(), // 'win' | 'loss' | 'draw'
  // JSON array of column indices (0-6) representing each move
  moves: text('moves').notNull().$type<string>(),
  // Total number of moves in the game
  moveCount: integer('move_count').notNull(),
  // ELO rating change from this game
  ratingChange: integer('rating_change').default(0),
  // Type of opponent
  opponentType: text('opponent_type').notNull().default('ai'), // 'human' | 'ai'
  // Opponent user ID (for bot vs bot matchup tracking)
  opponentId: text('opponent_id'),
  // AI difficulty level (null for human games)
  aiDifficulty: text('ai_difficulty'), // 'beginner' | 'intermediate' | 'expert' | 'perfect'
  // Bot persona ID for ranked bot games
  botPersonaId: text('bot_persona_id').references(() => botPersonas.id),
  // Which player the user played as (1 = red/first, 2 = yellow/second)
  playerNumber: integer('player_number').notNull().default(1),
  createdAt: integer('created_at').notNull(),
}, (table) => [
  index('idx_games_user_id').on(table.userId),
  index('idx_games_created_at').on(table.createdAt),
  index('idx_games_outcome').on(table.outcome),
  index('idx_games_opponent_type').on(table.opponentType),
  index('idx_games_ai_difficulty').on(table.aiDifficulty),
  index('idx_games_bot_persona').on(table.botPersonaId),
  index('idx_games_opponent_id').on(table.opponentId),
  index('idx_games_user_opponent').on(table.userId, table.opponentId),
])

export const gamesRelations = relations(games, ({ one, many }) => ({
  user: one(users, {
    fields: [games.userId],
    references: [users.id],
  }),
  botPersona: one(botPersonas, {
    fields: [games.botPersonaId],
    references: [botPersonas.id],
  }),
  ratingHistory: many(ratingHistory),
}))

// =============================================================================
// Rating History Table
// =============================================================================

export const ratingHistory = sqliteTable('rating_history', {
  id: text('id').primaryKey(),
  userId: text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  gameId: text('game_id').notNull().references(() => games.id, { onDelete: 'cascade' }),
  ratingBefore: integer('rating_before').notNull(),
  ratingAfter: integer('rating_after').notNull(),
  ratingChange: integer('rating_change').notNull(),
  createdAt: integer('created_at').notNull(),
}, (table) => [
  index('idx_rating_history_user_id').on(table.userId),
  index('idx_rating_history_created_at').on(table.createdAt),
])

export const ratingHistoryRelations = relations(ratingHistory, ({ one }) => ({
  user: one(users, {
    fields: [ratingHistory.userId],
    references: [users.id],
  }),
  game: one(games, {
    fields: [ratingHistory.gameId],
    references: [games.id],
  }),
}))

// =============================================================================
// Matchmaking Queue Table
// =============================================================================

export const matchmakingQueue = sqliteTable('matchmaking_queue', {
  id: text('id').primaryKey(),
  userId: text('user_id').unique().notNull().references(() => users.id, { onDelete: 'cascade' }),
  rating: integer('rating').notNull(),
  mode: text('mode').notNull().default('ranked'), // 'ranked' | 'casual'
  initialTolerance: integer('initial_tolerance').notNull().default(100),
  spectatable: integer('spectatable').notNull().default(1),
  joinedAt: integer('joined_at').notNull(),
}, (table) => [
  index('idx_matchmaking_queue_rating').on(table.rating),
  index('idx_matchmaking_queue_mode').on(table.mode),
  index('idx_matchmaking_queue_joined_at').on(table.joinedAt),
])

export const matchmakingQueueRelations = relations(matchmakingQueue, ({ one }) => ({
  user: one(users, {
    fields: [matchmakingQueue.userId],
    references: [users.id],
  }),
}))

// =============================================================================
// Active Games Table
// =============================================================================

export const activeGames = sqliteTable('active_games', {
  id: text('id').primaryKey(),
  player1Id: text('player1_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  player2Id: text('player2_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  // JSON array of column indices (0-6) representing each move
  moves: text('moves').notNull().default('[]').$type<string>(),
  // Current turn: 1 or 2
  currentTurn: integer('current_turn').notNull().default(1),
  // Game status
  status: text('status').notNull().default('active'), // 'active' | 'completed' | 'abandoned'
  // Game mode (affects ELO)
  mode: text('mode').notNull().default('ranked'), // 'ranked' | 'casual'
  // Winner (null if ongoing, '1', '2', or 'draw')
  winner: text('winner'),
  // Rating snapshots at game start
  player1Rating: integer('player1_rating').notNull(),
  player2Rating: integer('player2_rating').notNull(),
  // Spectator support
  spectatable: integer('spectatable').notNull().default(1),
  spectatorCount: integer('spectator_count').notNull().default(0),
  // Activity tracking
  lastMoveAt: integer('last_move_at').notNull(),
  // Timer fields
  timeControlMs: integer('time_control_ms'),
  player1TimeMs: integer('player1_time_ms'),
  player2TimeMs: integer('player2_time_ms'),
  turnStartedAt: integer('turn_started_at'),
  // Bot game fields
  isBotGame: integer('is_bot_game').notNull().default(0),
  botDifficulty: text('bot_difficulty'),
  // Bot vs Bot fields
  isBotVsBot: integer('is_bot_vs_bot').notNull().default(0),
  bot1PersonaId: text('bot1_persona_id'),
  bot2PersonaId: text('bot2_persona_id'),
  moveDelayMs: integer('move_delay_ms'),
  nextMoveAt: integer('next_move_at'),
  createdAt: integer('created_at').notNull(),
  updatedAt: integer('updated_at').notNull(),
}, (table) => [
  index('idx_active_games_player1').on(table.player1Id),
  index('idx_active_games_player2').on(table.player2Id),
  index('idx_active_games_status').on(table.status),
  index('idx_active_games_updated_at').on(table.updatedAt),
  index('idx_active_games_spectatable').on(table.spectatable, table.status),
  index('idx_active_games_timed').on(table.status, table.timeControlMs, table.turnStartedAt),
  index('idx_active_games_bot').on(table.isBotGame, table.status),
  index('idx_active_games_bot_vs_bot').on(table.isBotVsBot, table.status, table.nextMoveAt),
])

export const activeGamesRelations = relations(activeGames, ({ one, many }) => ({
  player1: one(users, {
    fields: [activeGames.player1Id],
    references: [users.id],
    relationName: 'player1',
  }),
  player2: one(users, {
    fields: [activeGames.player2Id],
    references: [users.id],
    relationName: 'player2',
  }),
  messages: many(gameMessages),
}))

// =============================================================================
// Game Messages Table
// =============================================================================

export const gameMessages = sqliteTable('game_messages', {
  id: text('id').primaryKey(),
  gameId: text('game_id').notNull().references(() => activeGames.id, { onDelete: 'cascade' }),
  senderId: text('sender_id').notNull(),
  senderType: text('sender_type').notNull(), // 'human' | 'bot'
  content: text('content').notNull(),
  createdAt: integer('created_at').notNull(),
}, (table) => [
  index('idx_game_messages_game_id').on(table.gameId),
  index('idx_game_messages_created_at').on(table.createdAt),
])

export const gameMessagesRelations = relations(gameMessages, ({ one }) => ({
  game: one(activeGames, {
    fields: [gameMessages.gameId],
    references: [activeGames.id],
  }),
}))

// =============================================================================
// Bot Personas Table
// =============================================================================

export const botPersonas = sqliteTable('bot_personas', {
  id: text('id').primaryKey(),
  name: text('name').notNull(),
  description: text('description').notNull(),
  avatarUrl: text('avatar_url'),
  aiEngine: text('ai_engine').notNull().default('minimax'),
  aiConfig: text('ai_config').notNull().default('{}').$type<string>(),
  chatPersonality: text('chat_personality').notNull().default('{}').$type<string>(),
  playStyle: text('play_style').notNull().default('balanced'), // 'aggressive' | 'defensive' | 'balanced' | 'tricky' | 'adaptive'
  baseElo: integer('base_elo').notNull().default(1200),
  currentElo: integer('current_elo').notNull().default(1200),
  gamesPlayed: integer('games_played').notNull().default(0),
  wins: integer('wins').notNull().default(0),
  losses: integer('losses').notNull().default(0),
  draws: integer('draws').notNull().default(0),
  isActive: integer('is_active').notNull().default(1),
  createdAt: integer('created_at').notNull(),
  updatedAt: integer('updated_at').notNull(),
}, (table) => [
  index('idx_bot_personas_active').on(table.isActive, table.currentElo),
])

export const botPersonasRelations = relations(botPersonas, ({ many }) => ({
  users: many(users),
  games: many(games),
  playerBotStats: many(playerBotStats),
}))

// =============================================================================
// Player Bot Stats Table
// =============================================================================

export const playerBotStats = sqliteTable('player_bot_stats', {
  userId: text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  botPersonaId: text('bot_persona_id').notNull().references(() => botPersonas.id, { onDelete: 'cascade' }),
  wins: integer('wins').notNull().default(0),
  losses: integer('losses').notNull().default(0),
  draws: integer('draws').notNull().default(0),
  currentStreak: integer('current_streak').notNull().default(0),
  bestWinStreak: integer('best_win_streak').notNull().default(0),
  firstWinAt: integer('first_win_at'),
  lastPlayedAt: integer('last_played_at').notNull(),
}, (table) => [
  primaryKey({ columns: [table.userId, table.botPersonaId] }),
  index('idx_player_bot_stats_user').on(table.userId),
  index('idx_player_bot_stats_bot').on(table.botPersonaId),
  index('idx_player_bot_stats_last_played').on(table.lastPlayedAt),
])

export const playerBotStatsRelations = relations(playerBotStats, ({ one }) => ({
  user: one(users, {
    fields: [playerBotStats.userId],
    references: [users.id],
  }),
  botPersona: one(botPersonas, {
    fields: [playerBotStats.botPersonaId],
    references: [botPersonas.id],
  }),
}))

// =============================================================================
// Challenges Table
// =============================================================================

export const challenges = sqliteTable('challenges', {
  id: text('id').primaryKey(),
  challengerId: text('challenger_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  challengerUsername: text('challenger_username').notNull(),
  challengerRating: integer('challenger_rating').notNull(),
  targetId: text('target_id').references(() => users.id, { onDelete: 'set null' }),
  targetUsername: text('target_username').notNull(),
  targetRating: integer('target_rating'),
  status: text('status').notNull().default('pending'), // 'pending' | 'accepted' | 'cancelled' | 'declined' | 'expired'
  createdAt: integer('created_at').notNull(),
  expiresAt: integer('expires_at').notNull(),
  gameId: text('game_id').references(() => activeGames.id, { onDelete: 'set null' }),
}, (table) => [
  index('idx_challenges_challenger').on(table.challengerId, table.status),
  index('idx_challenges_target').on(table.targetId, table.status),
  index('idx_challenges_expires').on(table.expiresAt),
  index('idx_challenges_status').on(table.status, table.createdAt),
])

export const challengesRelations = relations(challenges, ({ one }) => ({
  challenger: one(users, {
    fields: [challenges.challengerId],
    references: [users.id],
    relationName: 'challenger',
  }),
  target: one(users, {
    fields: [challenges.targetId],
    references: [users.id],
    relationName: 'target',
  }),
  game: one(activeGames, {
    fields: [challenges.gameId],
    references: [activeGames.id],
  }),
}))
