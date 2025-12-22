/**
 * Shared polling interval constants for frontend hooks
 *
 * Centralizing these values makes it easier to tune UI responsiveness
 * and ensures consistency across all polling-based hooks.
 */

/**
 * Interval for polling game state updates (in milliseconds).
 * Used by game hooks and spectator views.
 */
export const GAME_POLL_INTERVAL = 500 // 500ms for responsive gameplay

/**
 * Interval for polling chat messages (in milliseconds).
 * Used by chat hooks in games and spectator views.
 */
export const CHAT_POLL_INTERVAL = 1000 // 1 second for responsive chat

/**
 * Minimum time to wait for bot response (in milliseconds).
 * This prevents the bot from responding too quickly, which can feel unnatural.
 */
export const BOT_MIN_RESPONSE_TIME_MS = 1000 // 1 second minimum
