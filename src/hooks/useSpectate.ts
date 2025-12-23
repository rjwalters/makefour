/**
 * Custom React hook for spectating live games
 *
 * Handles:
 * - Fetching list of live games
 * - Joining a game as spectator
 * - Polling for game state updates
 * - Leaving spectator mode
 */

import { useState, useCallback, useRef } from 'react'
import type { Board } from '../game/makefour'
import { usePolling, useIsMounted } from './usePolling'
import { GAME_POLL_INTERVAL, LIVE_GAMES_POLL_INTERVAL } from '../lib/pollingConstants'

export interface LiveGame {
  id: string
  player1: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  player2: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  moveCount: number
  currentTurn: 1 | 2
  mode: 'ranked' | 'casual'
  spectatorCount: number
  createdAt: number
  updatedAt: number
  // Bot vs bot specific fields
  isBotVsBot?: boolean
  nextMoveAt?: number | null
}

export interface SpectatorGameState {
  id: string
  player1: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  player2: {
    rating: number
    displayName: string
    isBot?: boolean
    personaId?: string
  }
  currentTurn: 1 | 2
  moves: number[]
  board: Board | null
  status: 'active' | 'completed' | 'abandoned'
  winner: '1' | '2' | 'draw' | null
  mode: 'ranked' | 'casual'
  spectatorCount: number
  lastMoveAt: number
  createdAt: number
  // Timer fields
  timeControlMs: number | null
  player1TimeMs: number | null
  player2TimeMs: number | null
  turnStartedAt: number | null
  // Bot vs bot specific fields
  isBotVsBot?: boolean
  moveDelayMs?: number | null
  nextMoveAt?: number | null
}

export type SpectatorStatus = 'idle' | 'browsing' | 'loading' | 'watching' | 'error'

interface SpectatorState {
  status: SpectatorStatus
  error: string | null
  liveGames: LiveGame[]
  totalGames: number
  currentGame: SpectatorGameState | null
  isLoading: boolean
}

export function useSpectate() {
  const isMounted = useIsMounted()

  const [state, setState] = useState<SpectatorState>({
    status: 'idle',
    error: null,
    liveGames: [],
    totalGames: 0,
    currentGame: null,
    isLoading: false,
  })

  // Track the current game ID for polling
  const currentGameIdRef = useRef<string | null>(null)

  /**
   * Fetch list of live games
   */
  const fetchLiveGames = useCallback(async (options?: {
    limit?: number
    offset?: number
    minRating?: number
    maxRating?: number
    mode?: 'ranked' | 'casual'
  }) => {
    try {
      const params = new URLSearchParams()
      if (options?.limit) params.set('limit', String(options.limit))
      if (options?.offset) params.set('offset', String(options.offset))
      if (options?.minRating) params.set('minRating', String(options.minRating))
      if (options?.maxRating) params.set('maxRating', String(options.maxRating))
      if (options?.mode) params.set('mode', options.mode)

      const url = `/api/games/live${params.toString() ? `?${params}` : ''}`
      const response = await fetch(url)

      if (!response.ok) {
        throw new Error('Failed to fetch live games')
      }

      const data = await response.json() as {
        games: LiveGame[]
        total: number
        limit: number
        offset: number
      }

      if (!isMounted.current) return

      setState((prev) => ({
        ...prev,
        liveGames: data.games,
        totalGames: data.total,
        error: null,
      }))

      return data
    } catch (error) {
      if (!isMounted.current) return
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to fetch live games',
      }))
    }
  }, [isMounted])

  /**
   * Start browsing live games
   */
  const startBrowsing = useCallback(async () => {
    setState((prev) => ({ ...prev, status: 'browsing', isLoading: true }))

    await fetchLiveGames()

    if (!isMounted.current) return

    setState((prev) => ({ ...prev, isLoading: false }))
    // Polling will start automatically via usePolling when status becomes 'browsing'
  }, [fetchLiveGames, isMounted])

  /**
   * Stop browsing live games
   */
  const stopBrowsing = useCallback(() => {
    currentGameIdRef.current = null
    setState({
      status: 'idle',
      error: null,
      liveGames: [],
      totalGames: 0,
      currentGame: null,
      isLoading: false,
    })
    // Polling will stop automatically via usePolling when status becomes 'idle'
  }, [])

  /**
   * Fetch game state for spectating
   */
  const fetchGameState = useCallback(async (gameId: string) => {
    try {
      const response = await fetch(`/api/games/${gameId}/spectate`)

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || 'Failed to fetch game')
      }

      const gameState = await response.json() as SpectatorGameState

      if (!isMounted.current) return null

      setState((prev) => ({
        ...prev,
        status: 'watching',
        currentGame: gameState,
        error: null,
      }))

      // Stop polling if game is over
      if (gameState.status !== 'active') {
        currentGameIdRef.current = null
      }

      return gameState
    } catch (error) {
      if (!isMounted.current) return null
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Failed to fetch game',
      }))
      return null
    }
  }, [isMounted])

  /**
   * Join a game as spectator
   */
  const watchGame = useCallback(async (gameId: string) => {
    setState((prev) => ({ ...prev, status: 'loading', isLoading: true }))

    const gameState = await fetchGameState(gameId)

    if (!isMounted.current) return

    setState((prev) => ({ ...prev, isLoading: false }))

    if (!gameState) return

    // Set the game ID for polling (if game is active)
    if (gameState.status === 'active') {
      currentGameIdRef.current = gameId
    }
    // Polling will start automatically via usePolling
  }, [fetchGameState, isMounted])

  /**
   * Leave spectator mode and return to browsing
   */
  const leaveGame = useCallback(() => {
    currentGameIdRef.current = null

    setState((prev) => ({
      ...prev,
      status: 'browsing',
      currentGame: null,
    }))

    // Fetch live games immediately, polling will resume via usePolling
    fetchLiveGames()
  }, [fetchLiveGames])

  /**
   * Reset to idle state
   */
  const reset = useCallback(() => {
    currentGameIdRef.current = null
    setState({
      status: 'idle',
      error: null,
      liveGames: [],
      totalGames: 0,
      currentGame: null,
      isLoading: false,
    })
    // Polling will stop automatically via usePolling when status becomes 'idle'
  }, [])

  // Poll for live games when browsing
  const pollLiveGames = useCallback(async () => {
    await fetchLiveGames()
  }, [fetchLiveGames])

  usePolling(pollLiveGames, {
    interval: LIVE_GAMES_POLL_INTERVAL,
    enabled: state.status === 'browsing',
    immediate: false, // Already fetched in startBrowsing
  })

  // Poll for game state when watching an active game
  const pollCurrentGame = useCallback(async () => {
    const gameId = currentGameIdRef.current
    if (gameId) {
      await fetchGameState(gameId)
    }
  }, [fetchGameState])

  usePolling(pollCurrentGame, {
    interval: GAME_POLL_INTERVAL,
    enabled: state.status === 'watching' && currentGameIdRef.current !== null,
    immediate: false, // Already fetched in watchGame
  })

  return {
    ...state,
    startBrowsing,
    stopBrowsing,
    watchGame,
    leaveGame,
    fetchLiveGames,
    reset,
  }
}
