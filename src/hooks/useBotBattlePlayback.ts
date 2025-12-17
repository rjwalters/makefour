/**
 * Custom React hook for bot vs bot game playback
 *
 * Handles:
 * - Generating new bot vs bot games on-demand
 * - Playing back moves with configurable timing
 * - Playback controls (pause, resume, speed, restart)
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { replayMoves, type Board } from '../game/makefour'

export interface BotInfo {
  id: string
  personaId: string
  name: string
  rating: number
  difficulty: string
  engine: string
}

export interface GeneratedGame {
  id: string
  moves: number[]
  winner: 'bot1' | 'bot2' | 'draw'
  bot1: BotInfo
  bot2: BotInfo
  moveDelayMs: number
  createdAt: number
}

export type PlaybackStatus = 'idle' | 'loading' | 'playing' | 'paused' | 'finished' | 'error'

export interface PlaybackState {
  status: PlaybackStatus
  error: string | null
  game: GeneratedGame | null
  currentMoveIndex: number
  board: Board | null
  currentPlayer: 1 | 2
  playbackSpeed: number // 1 = normal, 2 = 2x speed, 0.5 = half speed
}

const DEFAULT_MOVE_DELAY = 1500 // 1.5 seconds per move

export function useBotBattlePlayback() {
  const [state, setState] = useState<PlaybackState>({
    status: 'idle',
    error: null,
    game: null,
    currentMoveIndex: -1,
    board: null,
    currentPlayer: 1,
    playbackSpeed: 1,
  })

  const playbackTimerRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      if (playbackTimerRef.current) {
        clearTimeout(playbackTimerRef.current)
      }
    }
  }, [])

  /**
   * Generate a new bot vs bot game
   */
  const generateGame = useCallback(async () => {
    // Clear any existing playback timer
    if (playbackTimerRef.current) {
      clearTimeout(playbackTimerRef.current)
      playbackTimerRef.current = null
    }

    setState((prev) => ({
      ...prev,
      status: 'loading',
      error: null,
      game: null,
      currentMoveIndex: -1,
      board: null,
      currentPlayer: 1,
    }))

    try {
      const response = await fetch('/api/bot/vs-bot/generate', {
        method: 'POST',
      })

      if (!response.ok) {
        // Try to parse error as JSON, but handle non-JSON responses gracefully
        let errorMessage = `Server error (${response.status})`
        const responseText = await response.text()
        try {
          const data = JSON.parse(responseText)
          errorMessage = data.error || errorMessage
        } catch {
          // Response wasn't JSON (e.g., HTML error page)
          if (responseText.length > 0 && responseText.length < 200 && !responseText.startsWith('<')) {
            errorMessage = responseText
          }
        }
        throw new Error(errorMessage)
      }

      const data = await response.json() as { success: boolean; game: GeneratedGame }

      if (!isMountedRef.current) return null

      // Start with empty board
      setState((prev) => ({
        ...prev,
        status: 'playing',
        game: data.game,
        currentMoveIndex: -1,
        board: Array.from({ length: 6 }, () => Array(7).fill(null)),
        currentPlayer: 1,
        error: null,
      }))

      return data.game
    } catch (error) {
      if (!isMountedRef.current) return null

      setState((prev) => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'Failed to generate game',
      }))
      return null
    }
  }, [])

  /**
   * Advance to next move in playback
   */
  const advanceMove = useCallback(() => {
    setState((prev) => {
      if (!prev.game || prev.status !== 'playing') return prev

      const nextMoveIndex = prev.currentMoveIndex + 1

      // Check if we've reached the end
      if (nextMoveIndex >= prev.game.moves.length) {
        return {
          ...prev,
          status: 'finished',
        }
      }

      // Replay moves up to and including this one
      const movesToReplay = prev.game.moves.slice(0, nextMoveIndex + 1)
      const gameState = replayMoves(movesToReplay)

      if (!gameState) {
        return {
          ...prev,
          status: 'error',
          error: 'Invalid move sequence',
        }
      }

      return {
        ...prev,
        currentMoveIndex: nextMoveIndex,
        board: gameState.board,
        currentPlayer: gameState.currentPlayer,
      }
    })
  }, [])

  /**
   * Schedule next move in playback
   */
  const scheduleNextMove = useCallback(() => {
    setState((currentState) => {
      if (!currentState.game || currentState.status !== 'playing') return currentState

      const delay = (currentState.game.moveDelayMs || DEFAULT_MOVE_DELAY) / currentState.playbackSpeed

      // Clear any existing timer
      if (playbackTimerRef.current) {
        clearTimeout(playbackTimerRef.current)
      }

      playbackTimerRef.current = setTimeout(() => {
        advanceMove()
      }, delay)

      return currentState
    })
  }, [advanceMove])

  // Schedule next move whenever we're playing and move index changes
  useEffect(() => {
    if (state.status === 'playing' && state.game) {
      if (state.currentMoveIndex < state.game.moves.length - 1) {
        scheduleNextMove()
      } else if (state.currentMoveIndex === state.game.moves.length - 1) {
        // Last move played, transition to finished
        setState((prev) => ({ ...prev, status: 'finished' }))
      }
    }
  }, [state.status, state.currentMoveIndex, state.game, scheduleNextMove])

  /**
   * Pause playback
   */
  const pause = useCallback(() => {
    if (playbackTimerRef.current) {
      clearTimeout(playbackTimerRef.current)
      playbackTimerRef.current = null
    }
    setState((prev) => ({
      ...prev,
      status: prev.status === 'playing' ? 'paused' : prev.status,
    }))
  }, [])

  /**
   * Resume playback
   */
  const resume = useCallback(() => {
    setState((prev) => {
      if (prev.status !== 'paused' || !prev.game) return prev

      // If we haven't finished, resume playing
      if (prev.currentMoveIndex < prev.game.moves.length - 1) {
        return { ...prev, status: 'playing' }
      }
      return prev
    })
  }, [])

  /**
   * Restart playback from beginning
   */
  const restart = useCallback(() => {
    if (playbackTimerRef.current) {
      clearTimeout(playbackTimerRef.current)
      playbackTimerRef.current = null
    }

    setState((prev) => {
      if (!prev.game) return prev

      return {
        ...prev,
        status: 'playing',
        currentMoveIndex: -1,
        board: Array.from({ length: 6 }, () => Array(7).fill(null)),
        currentPlayer: 1,
      }
    })
  }, [])

  /**
   * Set playback speed (1 = normal, 2 = 2x, 0.5 = half)
   */
  const setSpeed = useCallback((speed: number) => {
    setState((prev) => ({
      ...prev,
      playbackSpeed: Math.max(0.25, Math.min(4, speed)),
    }))
  }, [])

  /**
   * Jump to a specific move
   */
  const jumpToMove = useCallback((moveIndex: number) => {
    setState((prev) => {
      if (!prev.game) return prev

      const targetIndex = Math.max(-1, Math.min(moveIndex, prev.game.moves.length - 1))

      // Pause if jumping
      if (playbackTimerRef.current) {
        clearTimeout(playbackTimerRef.current)
        playbackTimerRef.current = null
      }

      if (targetIndex < 0) {
        return {
          ...prev,
          status: 'paused',
          currentMoveIndex: -1,
          board: Array.from({ length: 6 }, () => Array(7).fill(null)),
          currentPlayer: 1,
        }
      }

      const movesToReplay = prev.game.moves.slice(0, targetIndex + 1)
      const gameState = replayMoves(movesToReplay)

      if (!gameState) {
        return prev
      }

      return {
        ...prev,
        status: targetIndex >= prev.game.moves.length - 1 ? 'finished' : 'paused',
        currentMoveIndex: targetIndex,
        board: gameState.board,
        currentPlayer: gameState.currentPlayer,
      }
    })
  }, [])

  /**
   * Reset to idle state
   */
  const reset = useCallback(() => {
    if (playbackTimerRef.current) {
      clearTimeout(playbackTimerRef.current)
      playbackTimerRef.current = null
    }

    setState({
      status: 'idle',
      error: null,
      game: null,
      currentMoveIndex: -1,
      board: null,
      currentPlayer: 1,
      playbackSpeed: 1,
    })
  }, [])

  /**
   * Watch another game (generate new and start playing)
   */
  const watchAnother = useCallback(async () => {
    const game = await generateGame()
    return game
  }, [generateGame])

  // Calculate derived state
  const totalMoves = state.game?.moves.length ?? 0
  const progress = totalMoves > 0 ? (state.currentMoveIndex + 1) / totalMoves : 0
  const isPlaying = state.status === 'playing'
  const isPaused = state.status === 'paused'
  const isFinished = state.status === 'finished'
  const isLoading = state.status === 'loading'

  // Determine winner info for display
  const getWinnerInfo = useCallback(() => {
    if (!state.game || state.status !== 'finished') return null

    if (state.game.winner === 'draw') {
      return { type: 'draw' as const, name: null }
    }

    const winner = state.game.winner === 'bot1' ? state.game.bot1 : state.game.bot2
    return { type: 'win' as const, name: winner.name, player: state.game.winner === 'bot1' ? 1 : 2 }
  }, [state.game, state.status])

  return {
    // State
    ...state,
    totalMoves,
    progress,
    isPlaying,
    isPaused,
    isFinished,
    isLoading,
    winnerInfo: getWinnerInfo(),

    // Actions
    generateGame,
    pause,
    resume,
    restart,
    setSpeed,
    jumpToMove,
    reset,
    watchAnother,
  }
}
