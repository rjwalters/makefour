/**
 * Custom React hook for managing ranked bot games
 *
 * Handles:
 * - Creating a new ranked bot game
 * - Polling for game state updates
 * - Submitting moves (bot responds automatically)
 * - Resigning games
 *
 * Key improvements:
 * - Uses request coordination to prevent polling during submissions
 * - Version-based conflict resolution to reject stale updates
 * - Board derived from moves (single source of truth)
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import { useRequestCoordinator } from './useRequestCoordinator'
import { useNeuralBot } from './useNeuralBot'
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

interface NeuralConfig {
  modelId: string
  temperature: number
}

interface BotGameHookState {
  status: BotGameStatus
  error: string | null
  game: BotGameState | null
  difficulty: BotDifficulty
  personaId: string | null
  neuralConfig: NeuralConfig | null
}

const GAME_POLL_INTERVAL = 500 // 500ms for responsive gameplay

export function useBotGame() {
  const { apiCall } = useAuthenticatedApi()
  const coordinator = useRequestCoordinator()
  const neuralBot = useNeuralBot()
  const [state, setState] = useState<BotGameHookState>({
    status: 'idle',
    error: null,
    game: null,
    difficulty: 'intermediate',
    personaId: null,
    neuralConfig: null,
  })

  const gamePollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)
  // Track version for conflict resolution (uses moves.length as version)
  const lastConfirmedVersionRef = useRef<number>(0)
  // Track optimistic move count to prevent polls from overwriting optimistic updates
  // When set, polls with moves.length <= this value will be rejected
  const optimisticMoveCountRef = useRef<number | null>(null)

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
      setState((prev) => ({ ...prev, status: 'creating', error: null, personaId, neuralConfig: null }))

      try {
        const response = await apiCall<{
          gameId: string
          playerNumber: 1 | 2
          difficulty: string
          personaId: string | null
          botRating: number
          botMovedFirst: boolean
          botMove?: number
          neuralConfig?: { modelId: string; temperature: number } | null
        }>('/api/bot/game', {
          method: 'POST',
          body: JSON.stringify({ personaId, playerColor }),
        })

        if (!isMountedRef.current) return

        // Store neural config and load model if this is a neural bot
        if (response.neuralConfig) {
          setState((prev) => ({ ...prev, neuralConfig: response.neuralConfig ?? null }))
          // Pre-load the neural model for faster inference
          await neuralBot.loadModel(response.neuralConfig.modelId)
        }

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
    [apiCall, neuralBot]
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
   * Respects coordinator.shouldPoll() to prevent racing with submissions
   */
  const startGamePolling = useCallback(
    (gameId: string) => {
      if (gamePollRef.current) {
        clearInterval(gamePollRef.current)
      }

      gamePollRef.current = setInterval(async () => {
        // Skip poll if we're in the middle of a submission
        if (!coordinator.shouldPoll()) {
          return
        }

        try {
          const response = await apiCall<BotGameState>(`/api/bot/game/${gameId}`)

          if (!isMountedRef.current) return

          // Version-based conflict resolution
          const responseVersion = response.moves.length

          setState((prev) => {
            // If we have an optimistic update pending, reject polls that don't have MORE moves
            // This prevents in-flight polls from overwriting our optimistic state
            const optimisticCount = optimisticMoveCountRef.current
            if (optimisticCount !== null && responseVersion <= optimisticCount) {
              return prev
            }

            // Also check against current state
            if (prev.game && prev.game.moves.length > responseVersion) {
              return prev
            }

            // Update confirmed version
            lastConfirmedVersionRef.current = responseVersion

            return {
              ...prev,
              status: response.status === 'completed' ? 'completed' : 'playing',
              game: response,
            }
          })

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
    [apiCall, coordinator]
  )

  /**
   * Submit a move (bot will respond automatically)
   * Optimistically renders human move, then delays before showing bot's response
   * Uses coordinator to prevent polling from overwriting optimistic state
   * For neural bots, computes bot move client-side for real ONNX inference
   */
  const submitMove = useCallback(
    async (column: number) => {
      if (!state.game || !state.game.board) return false

      // Capture game state for rollback BEFORE any mutations
      const rollbackMoves = state.game.moves
      const rollbackBoard = state.game.board
      const rollbackTurn = state.game.currentTurn
      const rollbackIsYourTurn = state.game.isYourTurn
      const gameId = state.game.id
      const neuralConfig = state.neuralConfig

      const startTime = Date.now()

      // Optimistically apply the human's move locally
      const gameState = replayMoves(rollbackMoves)
      if (!gameState) return false

      const afterHumanMove = applyMove(gameState, column)
      if (!afterHumanMove) return false

      // For neural bots, compute the bot's response client-side
      // This uses real ONNX inference instead of server-side simulation
      let clientBotMove: number | undefined
      if (neuralConfig && neuralBot.isReady && afterHumanMove.winner === null) {
        const botPlayer: 1 | 2 = state.game.playerNumber === 1 ? 2 : 1
        const result = await neuralBot.computeMove(
          afterHumanMove.board,
          botPlayer,
          [...rollbackMoves, column],
          neuralConfig.temperature
        )
        if (result) {
          clientBotMove = result.column
        }
      }

      // Pause polling BEFORE updating optimistic state to prevent race conditions
      // This ensures no poll responses can overwrite our optimistic update
      coordinator.pausePolling()

      // Update state immediately with the human's move
      const optimisticMoves = [...rollbackMoves, column]

      // Set optimistic move count to reject in-flight polls
      optimisticMoveCountRef.current = optimisticMoves.length

      // Check if user's move resulted in a win (for optimistic winner display)
      const optimisticWinner = afterHumanMove.winner !== null
        ? (afterHumanMove.winner === 'draw' ? 'draw' : String(afterHumanMove.winner))
        : null

      setState((prev) => ({
        ...prev,
        // If user won, mark as completed optimistically
        status: optimisticWinner ? 'completed' : prev.status,
        game: prev.game
          ? {
              ...prev.game,
              moves: optimisticMoves,
              board: afterHumanMove.board,
              currentTurn: (prev.game.currentTurn === 1 ? 2 : 1) as 1 | 2,
              isYourTurn: false,
              // Set winner optimistically so win highlights render immediately
              winner: optimisticWinner as '1' | '2' | 'draw' | null,
              status: optimisticWinner ? 'completed' : prev.game.status,
            }
          : null,
      }))

      // Use coordinator to wrap the submission - this prevents polling during the request
      // Wrap in try/finally to ensure we always resume polling
      try {
        return await coordinator.withSubmission(async () => {
          try {
            // Include pre-computed bot move for neural bots
            const requestBody: { column: number; botMove?: number } = { column }
            if (clientBotMove !== undefined) {
              requestBody.botMove = clientBotMove
            }

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
            }>(`/api/bot/game/${gameId}`, {
              method: 'POST',
              body: JSON.stringify(requestBody),
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

            // Update confirmed version and clear optimistic flag
            lastConfirmedVersionRef.current = response.moves.length
            optimisticMoveCountRef.current = null

            // Now update with the full server response (including bot's move if any)
            // If no bot move, preserve the optimistic board to prevent visual flash
            setState((prev) => ({
              ...prev,
              status: response.status === 'completed' ? 'completed' : 'playing',
              game: prev.game
                ? {
                    ...prev.game,
                    moves: response.moves,
                    // Only update board from server if bot made a move, otherwise keep optimistic board
                    board: botMadeMove ? response.board : prev.game.board,
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
            // Clear optimistic flag on error
            optimisticMoveCountRef.current = null
            // Revert optimistic update on error using captured values (not stale closure!)
            setState((prev) => ({
              ...prev,
              game: prev.game
                ? {
                    ...prev.game,
                    moves: rollbackMoves,
                    board: rollbackBoard,
                    currentTurn: rollbackTurn,
                    isYourTurn: rollbackIsYourTurn,
                  }
                : null,
            }))
            return false
          }
        })
      } finally {
        // Always resume polling after submission completes (success or failure)
        coordinator.resumePolling()
      }
    },
    [apiCall, state.game, state.neuralConfig, coordinator, neuralBot]
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
      neuralConfig: null,
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
