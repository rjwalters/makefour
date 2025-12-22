/**
 * Hook for fetching and managing bot personas
 */

import { useApi } from './useApi'
import { API_BOT } from '../lib/apiEndpoints'

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

interface BotPersonasResponse {
  personas: BotPersona[]
}

export function useBotPersonas(): UseBotPersonasResult {
  const { data, loading, error, refetch } = useApi<BotPersona[], BotPersonasResponse>(
    API_BOT.PERSONAS,
    {
      transform: (response) => response.personas,
      initialData: [],
    }
  )

  return {
    personas: data ?? [],
    isLoading: loading,
    error,
    refetch,
  }
}
