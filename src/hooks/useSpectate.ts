/**
 * Custom React hook for spectating live games
 *
 * Handles:
 * - Fetching list of live games
 * - Joining a game as spectator
 * - Polling for game state updates
 * - Leaving spectator mode
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import type { Board } from '../game/makefour'

export interface LiveGame {
  id: string
  player1: {
    rating: number
    displayName: string
  }
  player2: {
    rating: number
    displayName: string
  }
  moveCount: number
  currentTurn: 1 | 2
  mode: 'ranked' | 'casual'
  spectatorCount: number
  createdAt: number
  updatedAt: number
}

export interface SpectatorGameState {
  id: string
  player1: {
    rating: number
    displayName: string
  }
  player2: {
    rating: number
    displayName: string
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

const LIVE_GAMES_POLL_INTERVAL = 5000 // 5 seconds for list refresh
const GAME_POLL_INTERVAL = 500 // 500ms for game state (same as players)

export function useSpectate() {
  const [state, setState] = useState<SpectatorState>({
    status: 'idle',
    error: null,
    liveGames: [],
    totalGames: 0,
    currentGame: null,
    isLoading: false,
  })

  const liveGamesPollRef = useRef<NodeJS.Timeout | null>(null)
  const gamePollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      if (liveGamesPollRef.current) {
        clearInterval(liveGamesPollRef.current)
      }
      if (gamePollRef.current) {
        clearInterval(gamePollRef.current)
      }
    }
  }, [])

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

      if (!isMountedRef.current) return

      setState((prev) => ({
        ...prev,
        liveGames: data.games,
        totalGames: data.total,
        error: null,
      }))

      return data
    } catch (error) {
      if (!isMountedRef.current) return
      setState((prev) => ({
        ...prev,
        error: error instanceof Error ? error.message : 'Failed to fetch live games',
      }))
    }
  }, [])

  /**
   * Start browsing live games
   */
  const startBrowsing = useCallback(async () => {
    setState((prev) => ({ ...prev, status: 'browsing', isLoading: true }))

    await fetchLiveGames()

    if (!isMountedRef.current) return

    setState((prev) => ({ ...prev, isLoading: false }))

    // Start polling for live games updates
    if (liveGamesPollRef.current) {
      clearInterval(liveGamesPollRef.current)
    }
    liveGamesPollRef.current = setInterval(() => fetchLiveGames(), LIVE_GAMES_POLL_INTERVAL)
  }, [fetchLiveGames])

  /**
   * Stop browsing live games
   */
  const stopBrowsing = useCallback(() => {
    if (liveGamesPollRef.current) {
      clearInterval(liveGamesPollRef.current)
      liveGamesPollRef.current = null
    }
    setState({
      status: 'idle',
      error: null,
      liveGames: [],
      totalGames: 0,
      currentGame: null,
      isLoading: false,
    })
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

      if (!isMountedRef.current) return null

      setState((prev) => ({
        ...prev,
        status: 'watching',
        currentGame: gameState,
        error: null,
      }))

      return gameState
    } catch (error) {
      if (!isMountedRef.current) return null
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Failed to fetch game',
      }))
      return null
    }
  }, [])

  /**
   * Join a game as spectator
   */
  const watchGame = useCallback(async (gameId: string) => {
    // Stop live games polling
    if (liveGamesPollRef.current) {
      clearInterval(liveGamesPollRef.current)
      liveGamesPollRef.current = null
    }

    setState((prev) => ({ ...prev, status: 'loading', isLoading: true }))

    const gameState = await fetchGameState(gameId)

    if (!isMountedRef.current) return

    setState((prev) => ({ ...prev, isLoading: false }))

    if (!gameState) return

    // Start polling for game state updates if game is active
    if (gameState.status === 'active') {
      if (gamePollRef.current) {
        clearInterval(gamePollRef.current)
      }
      gamePollRef.current = setInterval(async () => {
        const updated = await fetchGameState(gameId)
        // Stop polling if game is over
        if (updated && updated.status !== 'active') {
          if (gamePollRef.current) {
            clearInterval(gamePollRef.current)
            gamePollRef.current = null
          }
        }
      }, GAME_POLL_INTERVAL)
    }
  }, [fetchGameState])

  /**
   * Leave spectator mode and return to browsing
   */
  const leaveGame = useCallback(() => {
    if (gamePollRef.current) {
      clearInterval(gamePollRef.current)
      gamePollRef.current = null
    }

    setState((prev) => ({
      ...prev,
      status: 'browsing',
      currentGame: null,
    }))

    // Resume live games polling
    fetchLiveGames()
    liveGamesPollRef.current = setInterval(() => fetchLiveGames(), LIVE_GAMES_POLL_INTERVAL)
  }, [fetchLiveGames])

  /**
   * Reset to idle state
   */
  const reset = useCallback(() => {
    if (liveGamesPollRef.current) {
      clearInterval(liveGamesPollRef.current)
      liveGamesPollRef.current = null
    }
    if (gamePollRef.current) {
      clearInterval(gamePollRef.current)
      gamePollRef.current = null
    }

    setState({
      status: 'idle',
      error: null,
      liveGames: [],
      totalGames: 0,
      currentGame: null,
      isLoading: false,
    })
  }, [])

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
