/**
 * Hook for fetching and managing bot personas
 */

import { useState, useEffect, useCallback } from 'react'

export interface BotPersona {
  id: string
  name: string
  description: string
  avatarUrl: string | null
  playStyle: string
  rating: number
  gamesPlayed: number
  wins: number
  losses: number
  draws: number
  winRate: number
}

interface UseBotPersonasResult {
  personas: BotPersona[]
  isLoading: boolean
  error: string | null
  refetch: () => void
}

export function useBotPersonas(): UseBotPersonasResult {
  const [personas, setPersonas] = useState<BotPersona[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchPersonas = useCallback(async () => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/bot/personas')
      if (!response.ok) {
        throw new Error('Failed to fetch bot personas')
      }
      const data = await response.json()
      setPersonas(data.personas)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPersonas()
  }, [fetchPersonas])

  return {
    personas,
    isLoading,
    error,
    refetch: fetchPersonas,
  }
}
