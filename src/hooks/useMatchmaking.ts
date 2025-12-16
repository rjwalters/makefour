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
import { makeMove as applyMove, replayMoves, type Board } from '../game/makefour'

// Minimum time before bot responds (ms) - makes the interaction feel more natural
const BOT_MIN_RESPONSE_TIME_MS = 1000

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
  opponentUsername: string | null
  isYourTurn: boolean
  lastMoveAt: number
  // Timer fields (null for untimed games)
  timeControlMs: number | null
  player1TimeMs: number | null
  player2TimeMs: number | null
  turnStartedAt: number | null
  // Bot game fields
  isBotGame: boolean
  botDifficulty: string | null
}

interface MatchmakingState {
  status: MatchmakingStatus
  error: string | null
  waitTime: number
  currentTolerance: number
  mode: MatchmakingMode
  game: OnlineGameState | null
  botMatchReady: boolean
  userRating: number
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
    botMatchReady: false,
    userRating: 1200,
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
      botMatchReady: false,
      userRating: 1200,
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
        rating?: number
        botMatchReady?: boolean
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
          botMatchReady: response.botMatchReady || false,
          userRating: response.rating || prev.userRating,
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
    async (gameId: string, isBotGame: boolean = false) => {
      try {
        const endpoint = isBotGame ? `/api/bot/game/${gameId}` : `/api/match/${gameId}`
        const response = await apiCall<OnlineGameState>(endpoint)

        if (!isMountedRef.current) return

        setState((prev) => ({
          ...prev,
          status: response.status === 'completed' ? 'completed' : 'playing',
          game: response,
        }))

        // Start game polling if game is active
        if (response.status === 'active') {
          startGamePolling(gameId, response.isBotGame)
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
    (gameId: string, isBotGame: boolean = false) => {
      if (gamePollRef.current) {
        clearInterval(gamePollRef.current)
      }

      const endpoint = isBotGame ? `/api/bot/game/${gameId}` : `/api/match/${gameId}`
      gamePollRef.current = setInterval(async () => {
        try {
          const response = await apiCall<OnlineGameState>(endpoint)

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
   * For bot games: optimistically renders human move, then delays before showing bot's response
   */
  const submitMove = useCallback(
    async (column: number) => {
      if (!state.game || !state.game.board) return false

      const isBotGame = state.game.isBotGame
      const startTime = Date.now()

      // For bot games, optimistically apply the human's move locally
      let optimisticMoves: number[] | null = null
      if (isBotGame) {
        const gameState = replayMoves(state.game.moves)
        if (gameState) {
          const afterHumanMove = applyMove(gameState, column)
          if (afterHumanMove) {
            optimisticMoves = [...state.game.moves, column]
            // Update state immediately with the human's move
            setState((prev) => ({
              ...prev,
              game: prev.game
                ? {
                    ...prev.game,
                    moves: optimisticMoves!,
                    board: afterHumanMove.board,
                    currentTurn: (prev.game.currentTurn === 1 ? 2 : 1) as 1 | 2,
                    isYourTurn: false,
                  }
                : null,
            }))
          }
        }
      }

      try {
        const endpoint = isBotGame
          ? `/api/bot/game/${state.game.id}`
          : `/api/match/${state.game.id}`
        const response = await apiCall<{
          success: boolean
          moves: number[]
          board: Board
          currentTurn: 1 | 2
          status: 'active' | 'completed'
          winner: '1' | '2' | 'draw' | null
          isYourTurn: boolean
          // Timer fields
          timeControlMs: number | null
          player1TimeMs: number | null
          player2TimeMs: number | null
          turnStartedAt: number | null
        }>(endpoint, {
          method: 'POST',
          body: JSON.stringify({ column }),
        })

        if (!isMountedRef.current) return false

        // For bot games, add delay before showing bot's response
        const botMadeMove = isBotGame && optimisticMoves && response.moves.length > optimisticMoves.length
        if (botMadeMove) {
          const elapsed = Date.now() - startTime
          const remainingDelay = Math.max(0, BOT_MIN_RESPONSE_TIME_MS - elapsed)
          if (remainingDelay > 0) {
            await new Promise(resolve => setTimeout(resolve, remainingDelay))
          }
          if (!isMountedRef.current) return false
        }

        // For bot games where we did optimistic update, only update what changed
        // to avoid re-rendering the human's piece
        if (botMadeMove) {
          // Apply only the bot's move to the current board state (which already has human's move)
          const botMoveColumn = response.moves[response.moves.length - 1]
          setState((prev) => {
            if (!prev.game || !prev.game.board) return prev

            // Replay current moves to get game state, then apply bot's move
            const currentGameState = replayMoves(prev.game.moves)
            if (!currentGameState) return prev

            const afterBotMove = applyMove(currentGameState, botMoveColumn)
            if (!afterBotMove) return prev

            return {
              ...prev,
              status: response.status === 'completed' ? 'completed' : 'playing',
              game: {
                ...prev.game,
                moves: response.moves,
                board: afterBotMove.board,
                currentTurn: response.currentTurn,
                status: response.status,
                winner: response.winner,
                isYourTurn: response.isYourTurn,
                timeControlMs: response.timeControlMs,
                player1TimeMs: response.player1TimeMs,
                player2TimeMs: response.player2TimeMs,
                turnStartedAt: response.turnStartedAt,
              },
            }
          })
        } else {
          // For non-bot games or when bot didn't move, use server response directly
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
        }

        return true
      } catch (error) {
        console.error('Move submission error:', error)
        // Revert optimistic update on error for bot games
        if (isBotGame && optimisticMoves) {
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
        }
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
   * Play against a bot immediately (skip human matchmaking)
   */
  const playBotNow = useCallback(async () => {
    // Stop queue polling
    if (queuePollRef.current) {
      clearInterval(queuePollRef.current)
      queuePollRef.current = null
    }

    try {
      const response = await apiCall<{
        status: 'matched'
        gameId: string
        playerNumber: 1 | 2
        opponent: { name: string; rating: number; isBot: boolean }
        mode: MatchmakingMode
      }>('/api/matchmaking/play-bot', { method: 'POST' })

      if (!isMountedRef.current) return

      setState((prev) => ({
        ...prev,
        status: 'matched',
      }))

      // Fetch initial game state and start game polling (this is a bot game)
      await fetchGameState(response.gameId, true)
    } catch (error) {
      if (!isMountedRef.current) return
      setState((prev) => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Failed to start bot game',
      }))
    }
  }, [apiCall, fetchGameState])

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
      botMatchReady: false,
      userRating: 1200,
    })
  }, [])

  return {
    ...state,
    joinQueue,
    leaveQueue,
    playBotNow,
    submitMove,
    resign,
    reset,
  }
}
