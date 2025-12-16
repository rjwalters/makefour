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
import { makeMove as applyMove, replayMoves, type Board } from '../game/makefour'

// Minimum time before bot responds (ms) - makes the interaction feel more natural
const BOT_MIN_RESPONSE_TIME_MS = 1000

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
  botPersonaId: string | null
}

interface BotGameHookState {
  status: BotGameStatus
  error: string | null
  game: BotGameState | null
  difficulty: BotDifficulty
  personaId: string | null
}

const GAME_POLL_INTERVAL = 500 // 500ms for responsive gameplay

export function useBotGame() {
  const { apiCall } = useAuthenticatedApi()
  const [state, setState] = useState<BotGameHookState>({
    status: 'idle',
    error: null,
    game: null,
    difficulty: 'intermediate',
    personaId: null,
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
   * Create a new ranked bot game with a specific persona
   */
  const createGameWithPersona = useCallback(
    async (personaId: string, playerColor: 1 | 2) => {
      setState((prev) => ({ ...prev, status: 'creating', error: null, personaId }))

      try {
        const response = await apiCall<{
          gameId: string
          playerNumber: 1 | 2
          difficulty: string
          personaId: string | null
          botRating: number
          botMovedFirst: boolean
          botMove?: number
        }>('/api/bot/game', {
          method: 'POST',
          body: JSON.stringify({ personaId, playerColor }),
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
   * Create a new ranked bot game with difficulty (legacy support)
   */
  const createGame = useCallback(
    async (difficulty: BotDifficulty, playerColor: 1 | 2) => {
      setState((prev) => ({ ...prev, status: 'creating', error: null, difficulty }))

      try {
        const response = await apiCall<{
          gameId: string
          playerNumber: 1 | 2
          difficulty: string
          personaId: string | null
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
   * Optimistically renders human move, then delays before showing bot's response
   */
  const submitMove = useCallback(
    async (column: number) => {
      if (!state.game || !state.game.board) return false

      const startTime = Date.now()

      // Optimistically apply the human's move locally
      const gameState = replayMoves(state.game.moves)
      if (!gameState) return false

      const afterHumanMove = applyMove(gameState, column)
      if (!afterHumanMove) return false

      // Update state immediately with the human's move
      const optimisticMoves = [...state.game.moves, column]
      setState((prev) => ({
        ...prev,
        game: prev.game
          ? {
              ...prev.game,
              moves: optimisticMoves,
              board: afterHumanMove.board,
              currentTurn: (prev.game.currentTurn === 1 ? 2 : 1) as 1 | 2,
              isYourTurn: false,
            }
          : null,
      }))

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

        // Check if bot made a move (response has more moves than our optimistic state)
        const botMadeMove = response.moves.length > optimisticMoves.length

        if (botMadeMove) {
          // Calculate how long we've waited and ensure minimum delay
          const elapsed = Date.now() - startTime
          const remainingDelay = Math.max(0, BOT_MIN_RESPONSE_TIME_MS - elapsed)

          if (remainingDelay > 0) {
            // Wait for the remaining delay before showing bot's move
            await new Promise(resolve => setTimeout(resolve, remainingDelay))
          }
        }

        if (!isMountedRef.current) return false

        // Now update with the full server response (including bot's move)
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
        // Revert optimistic update on error
        setState((prev) => ({
          ...prev,
          game: prev.game
            ? {
                ...prev.game,
                moves: state.game!.moves,
                board: state.game!.board,
                currentTurn: state.game!.currentTurn,
                isYourTurn: state.game!.isYourTurn,
              }
            : null,
        }))
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
      personaId: null,
    })
  }, [])

  return {
    ...state,
    createGame,
    createGameWithPersona,
    submitMove,
    reset,
  }
}
