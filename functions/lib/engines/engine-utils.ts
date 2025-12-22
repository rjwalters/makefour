/**
 * Shared Engine Utilities
 *
 * Common functions used across multiple AI engines to avoid duplication.
 * Includes board evaluation, move generation, and move ordering.
 */

import { type Board, type Player, ROWS, COLUMNS, WIN_LENGTH } from '../game'

// ============================================================================
// EVALUATION WEIGHTS
// ============================================================================

/**
 * Weights for position evaluation.
 * Can be customized per engine for different play styles.
 */
export interface EvalWeights {
  win: number
  threeInRow: number
  twoInRow: number
  centerControl: number
  /** Optional: weight for opponent threats (defaults to same as own) */
  opponentThreats?: number
  /** Optional: bonus for double threats / forks */
  doubleThreats?: number
}

/**
 * Default evaluation weights for balanced play.
 */
export const DEFAULT_EVAL_WEIGHTS: EvalWeights = {
  win: 100000,
  threeInRow: 100,
  twoInRow: 10,
  centerControl: 3,
}

/**
 * Aggressive weights that prioritize offense over defense.
 */
export const AGGRESSIVE_EVAL_WEIGHTS: EvalWeights = {
  win: 100000,
  threeInRow: 150, // Higher own threats
  twoInRow: 15,
  centerControl: 2,
  opponentThreats: 80, // Lower priority on blocking
  doubleThreats: 500,
}

// ============================================================================
// MOVE GENERATION
// ============================================================================

/**
 * Returns valid column indices for the current board.
 * A column is valid if it has at least one empty cell (top row is empty).
 */
export function getValidMoves(board: Board): number[] {
  const moves: number[] = []
  for (let col = 0; col < COLUMNS; col++) {
    if (board[0][col] === null) {
      moves.push(col)
    }
  }
  return moves
}

/**
 * Orders moves with center columns first for better alpha-beta pruning.
 * Center moves are generally stronger in Connect Four.
 */
export function orderMoves(moves: number[]): number[] {
  const centerCol = Math.floor(COLUMNS / 2)
  return [...moves].sort((a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol))
}

// ============================================================================
// WINDOW EVALUATION
// ============================================================================

/**
 * Evaluates a window of 4 cells for scoring potential.
 *
 * @param window - Array of 4 cells to evaluate
 * @param player - The player to evaluate for
 * @param weights - Evaluation weights to use
 * @returns Score for this window from player's perspective
 */
export function evaluateWindow(
  window: (Player | null)[],
  player: Player,
  weights: EvalWeights = DEFAULT_EVAL_WEIGHTS
): number {
  const opponent: Player = player === 1 ? 2 : 1
  const playerCount = window.filter((c) => c === player).length
  const opponentCount = window.filter((c) => c === opponent).length
  const emptyCount = window.filter((c) => c === null).length

  // Mixed windows (both players have pieces) are worthless
  if (opponentCount > 0 && playerCount > 0) return 0

  // Own pieces
  if (playerCount === 4) return weights.win
  if (playerCount === 3 && emptyCount === 1) return weights.threeInRow
  if (playerCount === 2 && emptyCount === 2) return weights.twoInRow

  // Opponent pieces
  const oppThreats = weights.opponentThreats ?? weights.threeInRow
  const oppTwoWeight = weights.opponentThreats ? weights.twoInRow * 0.5 : weights.twoInRow

  if (opponentCount === 4) return -weights.win
  if (opponentCount === 3 && emptyCount === 1) return -oppThreats
  if (opponentCount === 2 && emptyCount === 2) return -oppTwoWeight

  return 0
}

// ============================================================================
// POSITION EVALUATION
// ============================================================================

/**
 * Iterates over all windows on the board and applies a callback.
 * A window is a group of 4 consecutive cells (horizontal, vertical, or diagonal).
 */
export function forEachWindow(
  board: Board,
  callback: (window: (Player | null)[]) => void
): void {
  // Horizontal windows
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      callback([
        board[row][col],
        board[row][col + 1],
        board[row][col + 2],
        board[row][col + 3],
      ])
    }
  }

  // Vertical windows
  for (let col = 0; col < COLUMNS; col++) {
    for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
      callback([
        board[row][col],
        board[row + 1][col],
        board[row + 2][col],
        board[row + 3][col],
      ])
    }
  }

  // Diagonal windows (down-right)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      callback([
        board[row][col],
        board[row + 1][col + 1],
        board[row + 2][col + 2],
        board[row + 3][col + 3],
      ])
    }
  }

  // Diagonal windows (down-left)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = WIN_LENGTH - 1; col < COLUMNS; col++) {
      callback([
        board[row][col],
        board[row + 1][col - 1],
        board[row + 2][col - 2],
        board[row + 3][col - 3],
      ])
    }
  }
}

/**
 * Evaluates the board position from the perspective of the given player.
 *
 * @param board - Current board state
 * @param player - Player to evaluate for
 * @param weights - Evaluation weights to use
 * @returns Score from player's perspective (positive = good for player)
 */
export function evaluatePosition(
  board: Board,
  player: Player,
  weights: EvalWeights = DEFAULT_EVAL_WEIGHTS
): number {
  let score = 0

  // Center column control
  const centerCol = Math.floor(COLUMNS / 2)
  for (let row = 0; row < ROWS; row++) {
    if (board[row][centerCol] === player) {
      score += weights.centerControl
    } else if (board[row][centerCol] !== null) {
      score -= weights.centerControl
    }
  }

  // Evaluate all windows
  forEachWindow(board, (window) => {
    score += evaluateWindow(window, player, weights)
  })

  return score
}

/**
 * Counts immediate threats (3-in-a-row with one empty) for a player.
 */
export function countThreats(board: Board, player: Player): number {
  let threats = 0

  forEachWindow(board, (window) => {
    const playerCount = window.filter((c) => c === player).length
    const emptyCount = window.filter((c) => c === null).length
    if (playerCount === 3 && emptyCount === 1) {
      threats++
    }
  })

  return threats
}

/**
 * Checks if a player has a double threat (fork) - two or more threats at once.
 */
export function hasDoubleThreat(board: Board, player: Player): boolean {
  return countThreats(board, player) >= 2
}
