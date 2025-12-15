/**
 * Preferences Context
 *
 * Provides centralized user preferences that sync with the server for
 * authenticated users and fall back to localStorage for guests.
 */

import type React from 'react'
import { createContext, useContext } from 'react'
import { usePreferences, type UserPreferences, DEFAULT_PREFERENCES } from '../hooks/usePreferences'

interface PreferencesContextType {
  preferences: UserPreferences
  updatePreferences: (updates: Partial<UserPreferences>) => void
  reload: () => Promise<void>
  isLoading: boolean
  isSyncing: boolean
  error: string | null
}

const PreferencesContext = createContext<PreferencesContextType | undefined>(undefined)

export function PreferencesProvider({ children }: { children: React.ReactNode }) {
  const {
    preferences,
    updatePreferences,
    reload,
    isLoading,
    isSyncing,
    error,
  } = usePreferences()

  return (
    <PreferencesContext.Provider
      value={{
        preferences,
        updatePreferences,
        reload,
        isLoading,
        isSyncing,
        error,
      }}
    >
      {children}
    </PreferencesContext.Provider>
  )
}

export function usePreferencesContext() {
  const context = useContext(PreferencesContext)
  if (context === undefined) {
    throw new Error('usePreferencesContext must be used within a PreferencesProvider')
  }
  return context
}

export { DEFAULT_PREFERENCES }
export type { UserPreferences }
