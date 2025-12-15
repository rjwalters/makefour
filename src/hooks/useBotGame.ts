/**
 * Custom React hook for managing ranked bot games
 *
 * Handles:
 * - Creating a new ranked bot game
 * - Polling for game state updates
 * - Submitting moves (bot responds automatically)
 * - Resigning games
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import type { Board } from '../game/makefour'

export type BotDifficulty = 'beginner' | 'intermediate' | 'expert' | 'perfect'

export type BotGameStatus =
  | 'idle'
  | 'creating'
  | 'playing'
  | 'completed'
  | 'error'

export interface BotGameState {
  id: string
  playerNumber: 1 | 2
  currentTurn: 1 | 2
  moves: number[]
  board: Board | null
  status: 'active' | 'completed' | 'abandoned'
  winner: '1' | '2' | 'draw' | null
  opponentRating: number
  isYourTurn: boolean
  lastMoveAt: number
  // Timer fields
  timeControlMs: number | null
  player1TimeMs: number | null
  player2TimeMs: number | null
  turnStartedAt: number | null
  // Bot info
  isBotGame: boolean
  botDifficulty: string | null
}

interface BotGameHookState {
  status: BotGameStatus
  error: string | null
  game: BotGameState | null
  difficulty: BotDifficulty
}

const GAME_POLL_INTERVAL = 500 // 500ms for responsive gameplay

export function useBotGame() {
  const { apiCall } = useAuthenticatedApi()
  const [state, setState] = useState<BotGameHookState>({
    status: 'idle',
    error: null,
    game: null,
    difficulty: 'intermediate',
  })

  const gamePollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      if (gamePollRef.current) {
        clearInterval(gamePollRef.current)
      }
    }
  }, [])

  /**
   * Create a new ranked bot game
   */
  const createGame = useCallback(
    async (difficulty: BotDifficulty, playerColor: 1 | 2) => {
      setState((prev) => ({ ...prev, status: 'creating', error: null, difficulty }))

      try {
        const response = await apiCall<{
          gameId: string
          playerNumber: 1 | 2
          difficulty: string
          botRating: number
          botMovedFirst: boolean
          botMove?: number
        }>('/api/bot/game', {
          method: 'POST',
          body: JSON.stringify({ difficulty, playerColor }),
        })

        if (!isMountedRef.current) return

        // Fetch initial game state
        await fetchGameState(response.gameId)
      } catch (error) {
        if (!isMountedRef.current) return
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: error instanceof Error ? error.message : 'Failed to create game',
        }))
      }
    },
    [apiCall]
  )

  /**
   * Fetch game state
   */
  const fetchGameState = useCallback(
    async (gameId: string) => {
      try {
        const response = await apiCall<BotGameState>(`/api/bot/game/${gameId}`)

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
          const response = await apiCall<BotGameState>(`/api/bot/game/${gameId}`)

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
          console.error('Bot game poll error:', error)
        }
      }, GAME_POLL_INTERVAL)
    },
    [apiCall]
  )

  /**
   * Submit a move (bot will respond automatically)
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
          timeControlMs: number | null
          player1TimeMs: number | null
          player2TimeMs: number | null
          turnStartedAt: number | null
        }>(`/api/bot/game/${state.game.id}`, {
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
                timeControlMs: response.timeControlMs,
                player1TimeMs: response.player1TimeMs,
                player2TimeMs: response.player2TimeMs,
                turnStartedAt: response.turnStartedAt,
              }
            : null,
        }))

        return true
      } catch (error) {
        console.error('Bot game move submission error:', error)
        return false
      }
    },
    [apiCall, state.game]
  )

  /**
   * Reset to idle state
   */
  const reset = useCallback(() => {
    if (gamePollRef.current) {
      clearInterval(gamePollRef.current)
      gamePollRef.current = null
    }

    setState({
      status: 'idle',
      error: null,
      game: null,
      difficulty: 'intermediate',
    })
  }, [])

  return {
    ...state,
    createGame,
    submitMove,
    reset,
  }
}
