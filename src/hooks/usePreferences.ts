/**
 * Custom React hook for managing user preferences with server sync
 *
 * Provides centralized preference management that:
 * - Loads from server for authenticated users
 * - Falls back to localStorage for guests
 * - Syncs localStorage → server on login
 * - Debounces saves to avoid excessive API calls
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import { STORAGE_KEY_PREFERENCES } from '../lib/storageKeys'
import { API_PREFERENCES } from '../lib/apiEndpoints'

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

export const DEFAULT_PREFERENCES: UserPreferences = {
  soundEnabled: false,
  soundVolume: 50,
  defaultGameMode: 'ai',
  defaultDifficulty: 'intermediate',
  defaultPlayerColor: 1,
  defaultMatchmakingMode: 'ranked',
  allowSpectators: true,
  theme: 'system',
}

const DEBOUNCE_MS = 1000

/**
 * Load preferences from localStorage
 */
function loadLocalPreferences(): UserPreferences {
  try {
    const saved = localStorage.getItem(STORAGE_KEY_PREFERENCES)
    if (saved) {
      const parsed = JSON.parse(saved)
      return mergeWithDefaults(parsed)
    }
  } catch {
    // Ignore parse errors
  }
  return { ...DEFAULT_PREFERENCES }
}

/**
 * Save preferences to localStorage
 */
function saveLocalPreferences(preferences: UserPreferences) {
  localStorage.setItem(STORAGE_KEY_PREFERENCES, JSON.stringify(preferences))
}

/**
 * Merge partial preferences with defaults
 */
function mergeWithDefaults(partial: Partial<UserPreferences>): UserPreferences {
  return {
    soundEnabled: typeof partial.soundEnabled === 'boolean' ? partial.soundEnabled : DEFAULT_PREFERENCES.soundEnabled,
    soundVolume: typeof partial.soundVolume === 'number' && partial.soundVolume >= 0 && partial.soundVolume <= 100
      ? partial.soundVolume
      : DEFAULT_PREFERENCES.soundVolume,
    defaultGameMode: partial.defaultGameMode && ['ai', 'hotseat', 'online'].includes(partial.defaultGameMode)
      ? partial.defaultGameMode
      : DEFAULT_PREFERENCES.defaultGameMode,
    defaultDifficulty: partial.defaultDifficulty && ['beginner', 'intermediate', 'expert', 'perfect'].includes(partial.defaultDifficulty)
      ? partial.defaultDifficulty
      : DEFAULT_PREFERENCES.defaultDifficulty,
    defaultPlayerColor: partial.defaultPlayerColor === 1 || partial.defaultPlayerColor === 2
      ? partial.defaultPlayerColor
      : DEFAULT_PREFERENCES.defaultPlayerColor,
    defaultMatchmakingMode: partial.defaultMatchmakingMode && ['ranked', 'casual'].includes(partial.defaultMatchmakingMode)
      ? partial.defaultMatchmakingMode
      : DEFAULT_PREFERENCES.defaultMatchmakingMode,
    allowSpectators: typeof partial.allowSpectators === 'boolean'
      ? partial.allowSpectators
      : DEFAULT_PREFERENCES.allowSpectators,
    theme: partial.theme && ['light', 'dark', 'system'].includes(partial.theme)
      ? partial.theme
      : DEFAULT_PREFERENCES.theme,
  }
}

export function usePreferences() {
  const { isAuthenticated } = useAuth()
  const { apiCall, getSessionToken } = useAuthenticatedApi()
  const [preferences, setPreferences] = useState<UserPreferences>(loadLocalPreferences)
  const [isLoading, setIsLoading] = useState(false)
  const [isSyncing, setIsSyncing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Refs for debouncing
  const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pendingUpdateRef = useRef<Partial<UserPreferences> | null>(null)
  const isInitialLoadRef = useRef(true)
  const previousAuthStateRef = useRef(isAuthenticated)

  /**
   * Save preferences to server (debounced)
   */
  const saveToServer = useCallback(async (prefs: UserPreferences) => {
    if (!getSessionToken()) return

    try {
      setIsSyncing(true)
      await apiCall(API_PREFERENCES, {
        method: 'PUT',
        body: JSON.stringify({ preferences: prefs }),
      })
      setError(null)
    } catch (err) {
      console.error('Failed to save preferences to server:', err)
      setError(err instanceof Error ? err.message : 'Failed to save preferences')
    } finally {
      setIsSyncing(false)
    }
  }, [apiCall, getSessionToken])

  /**
   * Load preferences from server
   */
  const loadFromServer = useCallback(async () => {
    if (!getSessionToken()) return null

    try {
      setIsLoading(true)
      const response = await apiCall<{ preferences: UserPreferences }>(API_PREFERENCES)
      setError(null)
      return response.preferences
    } catch (err) {
      console.error('Failed to load preferences from server:', err)
      setError(err instanceof Error ? err.message : 'Failed to load preferences')
      return null
    } finally {
      setIsLoading(false)
    }
  }, [apiCall, getSessionToken])

  /**
   * Sync localStorage to server on login
   * Strategy: Use localStorage values if they differ from defaults (user has customized)
   */
  const syncOnLogin = useCallback(async () => {
    const localPrefs = loadLocalPreferences()
    const serverPrefs = await loadFromServer()

    if (serverPrefs) {
      // Check if localStorage has customized preferences
      const localIsCustomized = JSON.stringify(localPrefs) !== JSON.stringify(DEFAULT_PREFERENCES)
      const serverIsDefault = JSON.stringify(serverPrefs) === JSON.stringify(DEFAULT_PREFERENCES)

      if (localIsCustomized && serverIsDefault) {
        // User has local customizations, server is default -> push local to server
        setPreferences(localPrefs)
        await saveToServer(localPrefs)
      } else {
        // Use server preferences (server wins)
        setPreferences(serverPrefs)
        saveLocalPreferences(serverPrefs)
      }
    }
  }, [loadFromServer, saveToServer])

  // Load preferences on mount and handle auth state changes
  useEffect(() => {
    const wasAuthenticated = previousAuthStateRef.current
    previousAuthStateRef.current = isAuthenticated

    if (isAuthenticated) {
      if (isInitialLoadRef.current || !wasAuthenticated) {
        // Initial load or just logged in
        isInitialLoadRef.current = false
        syncOnLogin()
      }
    } else {
      // Not authenticated, use localStorage
      setPreferences(loadLocalPreferences())
    }
  }, [isAuthenticated, syncOnLogin])

  /**
   * Update preferences (debounced for server sync)
   */
  const updatePreferences = useCallback((updates: Partial<UserPreferences>) => {
    setPreferences((current) => {
      const newPrefs = mergeWithDefaults({ ...current, ...updates })

      // Always save to localStorage immediately
      saveLocalPreferences(newPrefs)

      // Debounce server sync for authenticated users
      if (getSessionToken()) {
        if (debounceTimeoutRef.current) {
          clearTimeout(debounceTimeoutRef.current)
        }

        // Accumulate pending updates
        pendingUpdateRef.current = {
          ...pendingUpdateRef.current,
          ...updates,
        }

        debounceTimeoutRef.current = setTimeout(() => {
          if (pendingUpdateRef.current) {
            saveToServer(newPrefs)
            pendingUpdateRef.current = null
          }
        }, DEBOUNCE_MS)
      }

      return newPrefs
    })
  }, [getSessionToken, saveToServer])

  /**
   * Reload preferences from server
   */
  const reload = useCallback(async () => {
    if (isAuthenticated) {
      const serverPrefs = await loadFromServer()
      if (serverPrefs) {
        setPreferences(serverPrefs)
        saveLocalPreferences(serverPrefs)
      }
    } else {
      setPreferences(loadLocalPreferences())
    }
  }, [isAuthenticated, loadFromServer])

  // Cleanup debounce timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current)
      }
    }
  }, [])

  return {
    preferences,
    updatePreferences,
    reload,
    isLoading,
    isSyncing,
    error,
  }
}
