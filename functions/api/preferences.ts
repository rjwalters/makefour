import { validateSession, errorResponse, jsonResponse } from '../lib/auth'

interface Env {
  DB: D1Database
}

/**
 * User preferences schema
 */
export interface UserPreferences {
  // Sound settings
  soundEnabled: boolean
  soundVolume: number // 0-100

  // Game settings
  defaultGameMode: 'ai' | 'hotseat' | 'online'
  defaultDifficulty: 'beginner' | 'intermediate' | 'expert' | 'perfect'
  defaultPlayerColor: 1 | 2
  defaultMatchmakingMode: 'ranked' | 'casual'
  allowSpectators: boolean

  // Theme
  theme: 'light' | 'dark' | 'system'
}

const DEFAULT_PREFERENCES: UserPreferences = {
  soundEnabled: true,
  soundVolume: 50,
  defaultGameMode: 'ai',
  defaultDifficulty: 'intermediate',
  defaultPlayerColor: 1,
  defaultMatchmakingMode: 'ranked',
  allowSpectators: true,
  theme: 'system',
}

/**
 * Validates and merges preferences with defaults
 */
function parsePreferences(preferencesJson: string | null): UserPreferences {
  if (!preferencesJson) {
    return { ...DEFAULT_PREFERENCES }
  }

  try {
    const parsed = JSON.parse(preferencesJson)
    return {
      soundEnabled: typeof parsed.soundEnabled === 'boolean' ? parsed.soundEnabled : DEFAULT_PREFERENCES.soundEnabled,
      soundVolume: typeof parsed.soundVolume === 'number' && parsed.soundVolume >= 0 && parsed.soundVolume <= 100
        ? parsed.soundVolume
        : DEFAULT_PREFERENCES.soundVolume,
      defaultGameMode: ['ai', 'hotseat', 'online'].includes(parsed.defaultGameMode)
        ? parsed.defaultGameMode
        : DEFAULT_PREFERENCES.defaultGameMode,
      defaultDifficulty: ['beginner', 'intermediate', 'expert', 'perfect'].includes(parsed.defaultDifficulty)
        ? parsed.defaultDifficulty
        : DEFAULT_PREFERENCES.defaultDifficulty,
      defaultPlayerColor: parsed.defaultPlayerColor === 1 || parsed.defaultPlayerColor === 2
        ? parsed.defaultPlayerColor
        : DEFAULT_PREFERENCES.defaultPlayerColor,
      defaultMatchmakingMode: ['ranked', 'casual'].includes(parsed.defaultMatchmakingMode)
        ? parsed.defaultMatchmakingMode
        : DEFAULT_PREFERENCES.defaultMatchmakingMode,
      allowSpectators: typeof parsed.allowSpectators === 'boolean'
        ? parsed.allowSpectators
        : DEFAULT_PREFERENCES.allowSpectators,
      theme: ['light', 'dark', 'system'].includes(parsed.theme)
        ? parsed.theme
        : DEFAULT_PREFERENCES.theme,
    }
  } catch {
    return { ...DEFAULT_PREFERENCES }
  }
}

/**
 * GET /api/preferences - Fetch user preferences
 *
 * Returns the user's preferences merged with defaults.
 * Requires authentication.
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Validate session
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Get user preferences
    const user = await DB.prepare('SELECT preferences FROM users WHERE id = ?')
      .bind(session.userId)
      .first<{ preferences: string | null }>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const preferences = parsePreferences(user.preferences)

    return jsonResponse({ preferences })
  } catch (error) {
    console.error('Get preferences error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * PUT /api/preferences - Update user preferences
 *
 * Accepts a partial preferences object and merges with existing preferences.
 * Requires authentication.
 */
export async function onRequestPut(context: EventContext<Env, any, any>) {
  const { DB } = context.env

  try {
    // Validate session
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse request body
    let updates: Partial<UserPreferences>
    try {
      const body = await context.request.json()
      updates = body.preferences || body
    } catch {
      return errorResponse('Invalid JSON body', 400)
    }

    // Get current preferences
    const user = await DB.prepare('SELECT preferences FROM users WHERE id = ?')
      .bind(session.userId)
      .first<{ preferences: string | null }>()

    if (!user) {
      return errorResponse('User not found', 404)
    }

    const currentPreferences = parsePreferences(user.preferences)

    // Merge updates with validation
    const newPreferences: UserPreferences = {
      soundEnabled: typeof updates.soundEnabled === 'boolean'
        ? updates.soundEnabled
        : currentPreferences.soundEnabled,
      soundVolume: typeof updates.soundVolume === 'number' && updates.soundVolume >= 0 && updates.soundVolume <= 100
        ? updates.soundVolume
        : currentPreferences.soundVolume,
      defaultGameMode: updates.defaultGameMode && ['ai', 'hotseat', 'online'].includes(updates.defaultGameMode)
        ? updates.defaultGameMode
        : currentPreferences.defaultGameMode,
      defaultDifficulty: updates.defaultDifficulty && ['beginner', 'intermediate', 'expert', 'perfect'].includes(updates.defaultDifficulty)
        ? updates.defaultDifficulty
        : currentPreferences.defaultDifficulty,
      defaultPlayerColor: updates.defaultPlayerColor === 1 || updates.defaultPlayerColor === 2
        ? updates.defaultPlayerColor
        : currentPreferences.defaultPlayerColor,
      defaultMatchmakingMode: updates.defaultMatchmakingMode && ['ranked', 'casual'].includes(updates.defaultMatchmakingMode)
        ? updates.defaultMatchmakingMode
        : currentPreferences.defaultMatchmakingMode,
      allowSpectators: typeof updates.allowSpectators === 'boolean'
        ? updates.allowSpectators
        : currentPreferences.allowSpectators,
      theme: updates.theme && ['light', 'dark', 'system'].includes(updates.theme)
        ? updates.theme
        : currentPreferences.theme,
    }

    // Save to database
    await DB.prepare('UPDATE users SET preferences = ?, updated_at = ? WHERE id = ?')
      .bind(JSON.stringify(newPreferences), Date.now(), session.userId)
      .run()

    return jsonResponse({ preferences: newPreferences })
  } catch (error) {
    console.error('Update preferences error:', error)
    return errorResponse('Internal server error', 500)
  }
}
