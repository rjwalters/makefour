/**
 * Centralized API endpoint constants
 *
 * All API endpoints used throughout the application are defined here
 * to prevent typos, enable easy refactoring, and provide a single source
 * of truth for API paths.
 */

// =============================================================================
// Auth Endpoints
// =============================================================================

export const API_AUTH = {
  ME: '/api/auth/me',
  LOGIN: '/api/auth/login',
  REGISTER: '/api/auth/register',
  LOGOUT: '/api/auth/logout',
  GOOGLE: '/api/auth/google',
  RESEND_VERIFICATION: '/api/auth/resend-verification',
  VERIFY_EMAIL: '/api/auth/verify-email',
  FORGOT_PASSWORD: '/api/auth/forgot-password',
  RESET_PASSWORD: '/api/auth/reset-password',
} as const

// =============================================================================
// User Endpoints
// =============================================================================

export const API_USERS = {
  ME: '/api/users/me',
  ME_STATS: '/api/users/me/stats',
  ME_USERNAME: '/api/users/me/username',
  ME_PASSWORD: '/api/users/me/password',
  ME_BOT_STATS: '/api/users/me/bot-stats',
  meStatsHistory: (start: string, end: string) =>
    `/api/users/me/stats/history?start=${start}&end=${end}`,
  meBotStatsById: (botId: string) => `/api/users/me/bot-stats/${botId}`,
} as const

// =============================================================================
// Game Endpoints
// =============================================================================

export const API_GAMES = {
  BASE: '/api/games',
  LIVE: '/api/games/live',
  byId: (gameId: string) => `/api/games/${gameId}`,
  chat: (gameId: string) => `/api/games/${gameId}/chat`,
  chatSince: (gameId: string, since: string) =>
    `/api/games/${gameId}/chat?since=${since}`,
  spectate: (gameId: string) => `/api/games/${gameId}/spectate`,
  withParams: (limit: number, offset: number) =>
    `/api/games?limit=${limit}&offset=${offset}`,
} as const

// =============================================================================
// Match Endpoints (PvP)
// =============================================================================

export const API_MATCH = {
  byId: (gameId: string) => `/api/match/${gameId}`,
  chat: (gameId: string) => `/api/match/${gameId}/chat`,
  chatSince: (gameId: string, since: string) =>
    `/api/match/${gameId}/chat?since=${since}`,
  botReaction: (gameId: string) => `/api/match/${gameId}/bot-reaction`,
  resign: (gameId: string) => `/api/match/${gameId}/resign`,
} as const

// =============================================================================
// Matchmaking Endpoints
// =============================================================================

export const API_MATCHMAKING = {
  JOIN: '/api/matchmaking/join',
  LEAVE: '/api/matchmaking/leave',
  STATUS: '/api/matchmaking/status',
  PLAY_BOT: '/api/matchmaking/play-bot',
} as const

// =============================================================================
// Bot Endpoints
// =============================================================================

export const API_BOT = {
  PERSONAS: '/api/bot/personas',
  GAME: '/api/bot/game',
  VS_BOT_GENERATE: '/api/bot/vs-bot/generate',
  personaById: (id: string) => `/api/bot/personas/${id}`,
  gameById: (gameId: string) => `/api/bot/game/${gameId}`,
} as const

// =============================================================================
// Challenge Endpoints
// =============================================================================

export const API_CHALLENGES = {
  BASE: '/api/challenges',
  INCOMING: '/api/challenges/incoming',
  byId: (challengeId: string) => `/api/challenges/${challengeId}`,
  accept: (challengeId: string) => `/api/challenges/${challengeId}/accept`,
} as const

// =============================================================================
// Other Endpoints
// =============================================================================

export const API_PREFERENCES = '/api/preferences'

export const API_LEADERBOARD = {
  BASE: '/api/leaderboard',
  withParams: (limit: number, offset: number, includeBots: boolean) =>
    `/api/leaderboard?limit=${limit}&offset=${offset}&includeBots=${includeBots}`,
} as const

export const API_EXPORT = {
  GAMES: '/api/export/games',
} as const

export const API_MODELS = {
  BASE: '/api/models',
  ORACLE_V2_DATA: '/api/models/oracle-v2/data',
  CNN_ORACLE_V1_DATA: '/api/models/cnn-oracle-v1/data',
  RESNET_ORACLE_V1_DATA: '/api/models/resnet-oracle-v1/data',
} as const

// Debug endpoints (only used in development)
export const API_DEBUG = {
  LOGIN: '/api/debug/login',
  GAME: '/api/debug/game',
} as const
