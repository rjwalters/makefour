/**
 * Custom React hook for managing online matchmaking and game state
 *
 * Handles:
 * - Joining/leaving matchmaking queue
 * - Polling for match status
 * - Polling for game state updates
 * - Submitting moves
 * - Resigning games
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import type { Board } from '../game/makefour'

export type MatchmakingMode = 'ranked' | 'casual'

export type MatchmakingStatus =
  | 'idle'
  | 'joining'
  | 'queued'
  | 'matched'
  | 'playing'
  | 'completed'
  | 'error'

export interface OnlineGameState {
  id: string
  playerNumber: 1 | 2
  currentTurn: 1 | 2
  moves: number[]
  board: Board | null
  status: 'active' | 'completed' | 'abandoned'
  winner: '1' | '2' | 'draw' | null
  mode: MatchmakingMode
  opponentRating: number
  isYourTurn: boolean
  lastMoveAt: number
}

interface MatchmakingState {
  status: MatchmakingStatus
  error: string | null
  waitTime: number
  currentTolerance: number
  mode: MatchmakingMode
  game: OnlineGameState | null
}

const QUEUE_POLL_INTERVAL = 2000 // 2 seconds
const GAME_POLL_INTERVAL = 500 // 500ms for responsive gameplay

export function useMatchmaking() {
  const { apiCall } = useAuthenticatedApi()
  const [state, setState] = useState<MatchmakingState>({
    status: 'idle',
    error: null,
    waitTime: 0,
    currentTolerance: 100,
    mode: 'ranked',
    game: null,
  })

  const queuePollRef = useRef<NodeJS.Timeout | null>(null)
  const gamePollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      if (queuePollRef.current) {
        clearInterval(queuePollRef.current)
      }
      if (gamePollRef.current) {
        clearInterval(gamePollRef.current)
      }
    }
  }, [])

  /**
   * Join the matchmaking queue
   */
  const joinQueue = useCallback(
    async (mode: MatchmakingMode = 'ranked', spectatable: boolean = true) => {
      setState((prev) => ({ ...prev, status: 'joining', error: null, mode }))

      try {
        await apiCall('/api/matchmaking/join', {
          method: 'POST',
          body: JSON.stringify({ mode, spectatable }),
        })

        if (!isMountedRef.current) return

        setState((prev) => ({
          ...prev,
          status: 'queued',
          waitTime: 0,
        }))

        // Start polling for match status
        startQueuePolling()
      } catch (error) {
        if (!isMountedRef.current) return
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: error instanceof Error ? error.message : 'Failed to join queue',
        }))
      }
    },
    [apiCall]
  )

  /**
   * Leave the matchmaking queue
   */
  const leaveQueue = useCallback(async () => {
    // Stop polling first
    if (queuePollRef.current) {
      clearInterval(queuePollRef.current)
      queuePollRef.current = null
    }

    try {
      await apiCall('/api/matchmaking/leave', { method: 'POST' })
    } catch {
      // Ignore errors when leaving queue
    }

    if (!isMountedRef.current) return

    setState({
      status: 'idle',
      error: null,
      waitTime: 0,
      currentTolerance: 100,
      mode: 'ranked',
      game: null,
    })
  }, [apiCall])

  /**
   * Poll for matchmaking status
   */
  const pollQueueStatus = useCallback(async () => {
    try {
      const response = await apiCall<{
        status: 'queued' | 'matched' | 'not_queued'
        gameId?: string
        playerNumber?: 1 | 2
        waitTime?: number
        currentTolerance?: number
        mode?: MatchmakingMode
        opponent?: { rating: number }
      }>('/api/matchmaking/status')

      if (!isMountedRef.current) return

      if (response.status === 'matched' && response.gameId) {
        // Stop queue polling
        if (queuePollRef.current) {
          clearInterval(queuePollRef.current)
          queuePollRef.current = null
        }

        setState((prev) => ({
          ...prev,
          status: 'matched',
        }))

        // Fetch initial game state and start game polling
        await fetchGameState(response.gameId)
      } else if (response.status === 'queued') {
        setState((prev) => ({
          ...prev,
          waitTime: response.waitTime || 0,
          currentTolerance: response.currentTolerance || prev.currentTolerance,
        }))
      } else if (response.status === 'not_queued') {
        // User was removed from queue (shouldn't happen normally)
        if (queuePollRef.current) {
          clearInterval(queuePollRef.current)
          queuePollRef.current = null
        }
        setState((prev) => ({
          ...prev,
          status: 'idle',
          error: 'Removed from queue',
        }))
      }
    } catch (error) {
      // Don't set error state for transient polling failures
      console.error('Queue poll error:', error)
    }
  }, [apiCall])

  /**
   * Start polling for queue status
   */
  const startQueuePolling = useCallback(() => {
    if (queuePollRef.current) {
      clearInterval(queuePollRef.current)
    }

    // Poll immediately, then on interval
    pollQueueStatus()
    queuePollRef.current = setInterval(pollQueueStatus, QUEUE_POLL_INTERVAL)
  }, [pollQueueStatus])

  /**
   * Fetch game state
   */
  const fetchGameState = useCallback(
    async (gameId: string) => {
      try {
        const response = await apiCall<OnlineGameState>(`/api/match/${gameId}`)

        if (!isMountedRef.current) return

        setState((prev) => ({
          ...prev,
          status: response.status === 'completed' ? 'completed' : 'playing',
          game: response,
        }))

        // Start game polling if game is active
        if (response.status === 'active') {
          startGamePolling(gameId)
        } else {
          // Stop polling if game is over
          if (gamePollRef.current) {
            clearInterval(gamePollRef.current)
            gamePollRef.current = null
          }
        }
      } catch (error) {
        if (!isMountedRef.current) return
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: error instanceof Error ? error.message : 'Failed to fetch game',
        }))
      }
    },
    [apiCall]
  )

  /**
   * Start polling for game state
   */
  const startGamePolling = useCallback(
    (gameId: string) => {
      if (gamePollRef.current) {
        clearInterval(gamePollRef.current)
      }

      gamePollRef.current = setInterval(async () => {
        try {
          const response = await apiCall<OnlineGameState>(`/api/match/${gameId}`)

          if (!isMountedRef.current) return

          setState((prev) => ({
            ...prev,
            status: response.status === 'completed' ? 'completed' : 'playing',
            game: response,
          }))

          // Stop polling if game is over
          if (response.status !== 'active') {
            if (gamePollRef.current) {
              clearInterval(gamePollRef.current)
              gamePollRef.current = null
            }
          }
        } catch (error) {
          console.error('Game poll error:', error)
        }
      }, GAME_POLL_INTERVAL)
    },
    [apiCall]
  )

  /**
   * Submit a move
   */
  const submitMove = useCallback(
    async (column: number) => {
      if (!state.game) return false

      try {
        const response = await apiCall<{
          success: boolean
          moves: number[]
          board: Board
          currentTurn: 1 | 2
          status: 'active' | 'completed'
          winner: '1' | '2' | 'draw' | null
          isYourTurn: boolean
        }>(`/api/match/${state.game.id}`, {
          method: 'POST',
          body: JSON.stringify({ column }),
        })

        if (!isMountedRef.current) return false

        setState((prev) => ({
          ...prev,
          status: response.status === 'completed' ? 'completed' : 'playing',
          game: prev.game
            ? {
                ...prev.game,
                moves: response.moves,
                board: response.board,
                currentTurn: response.currentTurn,
                status: response.status,
                winner: response.winner,
                isYourTurn: response.isYourTurn,
              }
            : null,
        }))

        return true
      } catch (error) {
        console.error('Move submission error:', error)
        return false
      }
    },
    [apiCall, state.game]
  )

  /**
   * Resign from current game
   */
  const resign = useCallback(async () => {
    if (!state.game) return

    // Stop game polling
    if (gamePollRef.current) {
      clearInterval(gamePollRef.current)
      gamePollRef.current = null
    }

    try {
      const response = await apiCall<{
        success: boolean
        status: 'completed'
        winner: '1' | '2'
        resigned: boolean
      }>(`/api/match/${state.game.id}/resign`, { method: 'POST' })

      if (!isMountedRef.current) return

      setState((prev) => ({
        ...prev,
        status: 'completed',
        game: prev.game
          ? {
              ...prev.game,
              status: 'completed',
              winner: response.winner,
            }
          : null,
      }))
    } catch (error) {
      console.error('Resign error:', error)
    }
  }, [apiCall, state.game])

  /**
   * Reset to idle state
   */
  const reset = useCallback(() => {
    if (queuePollRef.current) {
      clearInterval(queuePollRef.current)
      queuePollRef.current = null
    }
    if (gamePollRef.current) {
      clearInterval(gamePollRef.current)
      gamePollRef.current = null
    }

    setState({
      status: 'idle',
      error: null,
      waitTime: 0,
      currentTolerance: 100,
      mode: 'ranked',
      game: null,
    })
  }, [])

  return {
    ...state,
    joinQueue,
    leaveQueue,
    submitMove,
    resign,
    reset,
  }
}
