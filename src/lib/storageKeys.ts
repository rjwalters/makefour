/**
 * Centralized localStorage key constants
 *
 * All localStorage keys used throughout the application are defined here
 * to prevent typos, enable easy refactoring, and provide a single source
 * of truth for storage key names.
 */

/**
 * Session token for authenticated API requests.
 * Stored after login, removed on logout.
 */
export const STORAGE_KEY_SESSION_TOKEN = 'makefour_session_token'

/**
 * Data Encryption Key for client-side encryption.
 * Used for encrypting sensitive data before sending to server.
 */
export const STORAGE_KEY_DEK = 'makefour_dek'

/**
 * User's theme preference (light/dark).
 * Persists theme choice across sessions.
 */
export const STORAGE_KEY_THEME = 'makefour_theme'

/**
 * Salt used for coach encryption.
 * Stored as JSON array of bytes.
 */
export const STORAGE_KEY_COACH_SALT = 'coach_salt'

/**
 * User preferences object.
 * Contains game settings, sound preferences, etc.
 */
export const STORAGE_KEY_PREFERENCES = 'makefour-preferences'
