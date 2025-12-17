/**
 * useGameState - Single source of truth for game state
 *
 * This hook manages game state using the move history as the canonical data.
 * Board, currentPlayer, and winner are derived from moves via replayMoves().
 *
 * Key features:
 * - Version-based conflict resolution (rejects stale updates)
 * - Optimistic updates with rollback support
 * - Derived state is memoized for performance
 */

import { useState, useMemo, useCallback, useRef } from 'react'
import {
  replayMoves,
  createGameState,
  makeMove,
  type GameState,
  type Board,
  type Player,
  type GameResult,
} from '../game/makefour'

export interface GameStateData {
  /** The canonical move history (single source of truth) */
  moves: number[]
  /** Monotonically increasing version for conflict resolution */
  version: number
  /** Pending optimistic version (null if confirmed) */
  pendingVersion: number | null
}

export interface UseGameStateReturn {
  // Canonical state
  moves: number[]
  version: number
  isOptimistic: boolean

  // Derived state (computed from moves)
  board: Board
  currentPlayer: Player
  winner: GameResult
  moveHistory: number[]

  // Actions
  /** Apply a move optimistically (increments version) */
  applyMove: (column: number) => boolean
  /** Set moves from server (with version check) */
  setMoves: (newMoves: number[], serverVersion?: number) => void
  /** Confirm optimistic state (clears pending flag) */
  confirmOptimistic: () => void
  /** Rollback to last confirmed state */
  rollback: () => void
  /** Reset to empty game */
  reset: () => void
}

export function useGameState(initialMoves: number[] = []): UseGameStateReturn {
  const [state, setState] = useState<GameStateData>(() => ({
    moves: initialMoves,
    version: initialMoves.length,
    pendingVersion: null,
  }))

  // Store last confirmed state for rollback
  const lastConfirmedRef = useRef<{ moves: number[]; version: number }>({
    moves: initialMoves,
    version: initialMoves.length,
  })

  // Derive full game state from moves (memoized)
  const gameState: GameState = useMemo(() => {
    if (state.moves.length === 0) {
      return createGameState()
    }
    const replayed = replayMoves(state.moves)
    // If replay fails (invalid moves), return empty state
    return replayed ?? createGameState()
  }, [state.moves])

  // Apply a move optimistically
  const applyMove = useCallback((column: number): boolean => {
    let success = false

    setState((prev) => {
      // Compute current state to validate move
      const currentState = prev.moves.length === 0
        ? createGameState()
        : replayMoves(prev.moves)

      if (!currentState) return prev

      // Try to apply the move
      const newState = makeMove(currentState, column)
      if (!newState) return prev

      success = true
      const newVersion = prev.version + 1

      return {
        moves: newState.moveHistory,
        version: newVersion,
        pendingVersion: newVersion,
      }
    })

    return success
  }, [])

  // Set moves from server response (with version-based conflict resolution)
  const setMoves = useCallback((newMoves: number[], serverVersion?: number) => {
    setState((prev) => {
      const incomingVersion = serverVersion ?? newMoves.length

      // Reject stale updates (version must be >= our current)
      // Exception: if we have a pending optimistic update, only accept
      // if incoming version matches or exceeds our optimistic state
      if (prev.pendingVersion !== null) {
        // We have an unconfirmed optimistic update
        if (incomingVersion < prev.pendingVersion) {
          // Server hasn't caught up to our optimistic state - ignore
          return prev
        }
        // Server caught up or is ahead - accept and confirm
        lastConfirmedRef.current = { moves: newMoves, version: incomingVersion }
        return {
          moves: newMoves,
          version: incomingVersion,
          pendingVersion: null,
        }
      }

      // No pending optimistic update - normal version check
      if (incomingVersion < prev.version) {
        return prev
      }

      lastConfirmedRef.current = { moves: newMoves, version: incomingVersion }
      return {
        moves: newMoves,
        version: incomingVersion,
        pendingVersion: null,
      }
    })
  }, [])

  // Confirm optimistic state (called when server acknowledges our move)
  const confirmOptimistic = useCallback(() => {
    setState((prev) => {
      if (prev.pendingVersion === null) return prev

      lastConfirmedRef.current = { moves: prev.moves, version: prev.version }
      return {
        ...prev,
        pendingVersion: null,
      }
    })
  }, [])

  // Rollback to last confirmed state (called on error)
  const rollback = useCallback(() => {
    setState({
      moves: lastConfirmedRef.current.moves,
      version: lastConfirmedRef.current.version,
      pendingVersion: null,
    })
  }, [])

  // Reset to empty game
  const reset = useCallback(() => {
    lastConfirmedRef.current = { moves: [], version: 0 }
    setState({
      moves: [],
      version: 0,
      pendingVersion: null,
    })
  }, [])

  return {
    // Canonical state
    moves: state.moves,
    version: state.version,
    isOptimistic: state.pendingVersion !== null,

    // Derived state
    board: gameState.board,
    currentPlayer: gameState.currentPlayer,
    winner: gameState.winner,
    moveHistory: gameState.moveHistory,

    // Actions
    applyMove,
    setMoves,
    confirmOptimistic,
    rollback,
    reset,
  }
}
