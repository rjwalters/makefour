/**
 * AI Coach Module for MakeFour
 *
 * This module provides AI-powered game analysis and move suggestions
 * using minimax search with alpha-beta pruning.
 *
 * Features:
 * - Depth-limited minimax search with alpha-beta pruning
 * - Position evaluation heuristics (center control, threats, connectivity)
 * - Configurable difficulty levels via search depth and error rate
 * - Move ranking and analysis
 */

import {
  type Board,
  type Player,
  getValidMoves,
  applyMove,
  checkWinner,
  ROWS,
  COLUMNS,
  WIN_LENGTH,
} from '../game/makefour'

// ============================================================================
// POSITION EVALUATION
// ============================================================================

/**
 * Weights for position evaluation heuristics.
 */
const EVAL_WEIGHTS = {
  WIN: 100000,
  THREE_IN_ROW: 100,
  TWO_IN_ROW: 10,
  CENTER_CONTROL: 3,
}

/**
 * Evaluates a window of 4 cells for scoring potential.
 * @param window - Array of 4 cells
 * @param player - The player to evaluate for
 * @returns Score for this window
 */
function evaluateWindow(window: (Player | null)[], player: Player): number {
  const opponent: Player = player === 1 ? 2 : 1
  const playerCount = window.filter((c) => c === player).length
  const opponentCount = window.filter((c) => c === opponent).length
  const emptyCount = window.filter((c) => c === null).length

  // If opponent has pieces in this window, we can't complete a four here
  if (opponentCount > 0 && playerCount > 0) {
    return 0
  }

  if (playerCount === 4) {
    return EVAL_WEIGHTS.WIN
  }
  if (playerCount === 3 && emptyCount === 1) {
    return EVAL_WEIGHTS.THREE_IN_ROW
  }
  if (playerCount === 2 && emptyCount === 2) {
    return EVAL_WEIGHTS.TWO_IN_ROW
  }

  // Opponent threats (negative score)
  if (opponentCount === 4) {
    return -EVAL_WEIGHTS.WIN
  }
  if (opponentCount === 3 && emptyCount === 1) {
    return -EVAL_WEIGHTS.THREE_IN_ROW
  }
  if (opponentCount === 2 && emptyCount === 2) {
    return -EVAL_WEIGHTS.TWO_IN_ROW
  }

  return 0
}

/**
 * Evaluates the board position from the perspective of the given player.
 * Considers center control, connectivity, and threats.
 *
 * @param board - The current board state
 * @param player - The player to evaluate for
 * @returns Numeric score (positive = good for player, negative = bad)
 */
export function evaluatePosition(board: Board, player: Player): number {
  let score = 0

  // Center column control - pieces in the center are more valuable
  const centerCol = Math.floor(COLUMNS / 2)
  for (let row = 0; row < ROWS; row++) {
    if (board[row][centerCol] === player) {
      score += EVAL_WEIGHTS.CENTER_CONTROL
    } else if (board[row][centerCol] !== null) {
      score -= EVAL_WEIGHTS.CENTER_CONTROL
    }
  }

  // Evaluate all horizontal windows
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      const window = [
        board[row][col],
        board[row][col + 1],
        board[row][col + 2],
        board[row][col + 3],
      ]
      score += evaluateWindow(window, player)
    }
  }

  // Evaluate all vertical windows
  for (let col = 0; col < COLUMNS; col++) {
    for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
      const window = [
        board[row][col],
        board[row + 1][col],
        board[row + 2][col],
        board[row + 3][col],
      ]
      score += evaluateWindow(window, player)
    }
  }

  // Evaluate positive diagonal windows (down-right)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = 0; col <= COLUMNS - WIN_LENGTH; col++) {
      const window = [
        board[row][col],
        board[row + 1][col + 1],
        board[row + 2][col + 2],
        board[row + 3][col + 3],
      ]
      score += evaluateWindow(window, player)
    }
  }

  // Evaluate negative diagonal windows (down-left)
  for (let row = 0; row <= ROWS - WIN_LENGTH; row++) {
    for (let col = WIN_LENGTH - 1; col < COLUMNS; col++) {
      const window = [
        board[row][col],
        board[row + 1][col - 1],
        board[row + 2][col - 2],
        board[row + 3][col - 3],
      ]
      score += evaluateWindow(window, player)
    }
  }

  return score
}

// ============================================================================
// MINIMAX WITH ALPHA-BETA PRUNING
// ============================================================================

/**
 * Result of a minimax search.
 */
interface MinimaxResult {
  score: number
  move: number | null
}

/**
 * Orders moves to improve alpha-beta pruning efficiency.
 * Center columns are searched first as they tend to be better moves.
 *
 * @param moves - Array of valid column indices
 * @returns Reordered array with center moves first
 */
function orderMoves(moves: number[]): number[] {
  const centerCol = Math.floor(COLUMNS / 2)
  return [...moves].sort((a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol))
}

/**
 * Minimax search with alpha-beta pruning.
 *
 * @param board - Current board state
 * @param depth - Remaining search depth
 * @param alpha - Alpha value for pruning (best score for maximizing player)
 * @param beta - Beta value for pruning (best score for minimizing player)
 * @param maximizingPlayer - Whether current player is maximizing
 * @param player - The player we're evaluating for (stays constant)
 * @param currentPlayer - The player whose turn it is at this node
 * @returns Object with score and best move
 */
export function minimax(
  board: Board,
  depth: number,
  alpha: number,
  beta: number,
  maximizingPlayer: boolean,
  player: Player,
  currentPlayer: Player
): MinimaxResult {
  // Check for terminal states
  const winner = checkWinner(board)
  if (winner !== null) {
    if (winner === 'draw') {
      return { score: 0, move: null }
    }
    // Winner found - return large score based on who won
    const winScore = EVAL_WEIGHTS.WIN + depth * 100 // Prefer faster wins
    return {
      score: winner === player ? winScore : -winScore,
      move: null,
    }
  }

  const validMoves = getValidMoves(board)
  if (validMoves.length === 0) {
    return { score: 0, move: null }
  }

  // Depth limit reached - use heuristic evaluation
  if (depth === 0) {
    return { score: evaluatePosition(board, player), move: null }
  }

  const orderedMoves = orderMoves(validMoves)
  const nextPlayer: Player = currentPlayer === 1 ? 2 : 1

  if (maximizingPlayer) {
    let maxScore = -Infinity
    let bestMove = orderedMoves[0]

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const { score } = minimax(
        result.board,
        depth - 1,
        alpha,
        beta,
        false,
        player,
        nextPlayer
      )

      if (score > maxScore) {
        maxScore = score
        bestMove = move
      }

      alpha = Math.max(alpha, score)
      if (beta <= alpha) {
        break // Beta cutoff
      }
    }

    return { score: maxScore, move: bestMove }
  } else {
    let minScore = Infinity
    let bestMove = orderedMoves[0]

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const { score } = minimax(
        result.board,
        depth - 1,
        alpha,
        beta,
        true,
        player,
        nextPlayer
      )

      if (score < minScore) {
        minScore = score
        bestMove = move
      }

      beta = Math.min(beta, score)
      if (beta <= alpha) {
        break // Alpha cutoff
      }
    }

    return { score: minScore, move: bestMove }
  }
}

/**
 * Finds the best move for the given position using minimax search.
 *
 * @param board - Current board state
 * @param player - The player to find a move for
 * @param depth - Search depth (higher = stronger but slower)
 * @returns Object with best move and evaluation score
 */
export function findBestMove(
  board: Board,
  player: Player,
  depth: number
): { move: number; score: number } {
  const result = minimax(board, depth, -Infinity, Infinity, true, player, player)
  return {
    move: result.move ?? getValidMoves(board)[0] ?? 0,
    score: result.score,
  }
}

/**
 * Represents a game position for analysis.
 */
export interface Position {
  board: Board
  currentPlayer: Player
  moveHistory: number[]
}

/**
 * Result of position analysis.
 */
export interface Analysis {
  /** Best move (column index 0-6) */
  bestMove: number
  /** Evaluation score (positive = good for current player) */
  score: number
  /** Human-readable explanation of the position */
  evaluation: string
  /** Whether this position is theoretically won/lost/drawn */
  theoreticalResult: 'win' | 'loss' | 'draw' | 'unknown'
  /** How confident the analysis is (0-1) */
  confidence: number
}

/**
 * Generates a human-readable evaluation description based on the score.
 */
function getEvaluationDescription(score: number): string {
  if (score >= EVAL_WEIGHTS.WIN) {
    return 'Winning position - forced win detected'
  }
  if (score <= -EVAL_WEIGHTS.WIN) {
    return 'Losing position - opponent has forced win'
  }
  if (score > 500) {
    return 'Strongly favorable position'
  }
  if (score > 100) {
    return 'Slightly favorable position'
  }
  if (score < -500) {
    return 'Strongly unfavorable position'
  }
  if (score < -100) {
    return 'Slightly unfavorable position'
  }
  return 'Roughly equal position'
}

/**
 * Determines the theoretical result based on the minimax score.
 */
function getTheoreticalResult(score: number): 'win' | 'loss' | 'draw' | 'unknown' {
  if (score >= EVAL_WEIGHTS.WIN) {
    return 'win'
  }
  if (score <= -EVAL_WEIGHTS.WIN) {
    return 'loss'
  }
  if (score === 0) {
    return 'draw'
  }
  return 'unknown'
}

/**
 * Analyzes a position and returns insights.
 * Uses minimax search with alpha-beta pruning for analysis.
 *
 * @param position - The current game position
 * @param difficulty - The difficulty level (determines search depth)
 * @returns Promise resolving to analysis results
 */
export async function analyzePosition(
  position: Position,
  difficulty: DifficultyLevel = 'intermediate'
): Promise<Analysis> {
  const validMoves = getValidMoves(position.board)

  if (validMoves.length === 0) {
    const winner = checkWinner(position.board)
    return {
      bestMove: -1,
      score: 0,
      evaluation: winner === 'draw' ? 'Game ended in a draw' : `Game is complete - Player ${winner} wins`,
      theoreticalResult: winner === 'draw' ? 'draw' : (winner === position.currentPlayer ? 'win' : 'loss'),
      confidence: 1,
    }
  }

  const config = DIFFICULTY_LEVELS[difficulty]
  const { move, score } = findBestMove(position.board, position.currentPlayer, config.searchDepth)

  // Confidence based on search depth (deeper = more confident)
  const confidence = Math.min(0.5 + config.searchDepth * 0.05, 1)

  return {
    bestMove: move,
    score,
    evaluation: getEvaluationDescription(score),
    theoreticalResult: getTheoreticalResult(score),
    confidence,
  }
}

/**
 * Suggests the best move for the current position.
 * Uses minimax search with alpha-beta pruning.
 *
 * @param position - The current game position
 * @param difficulty - The difficulty level (determines search depth and error rate)
 * @returns Promise resolving to the suggested column (0-6)
 */
export async function suggestMove(
  position: Position,
  difficulty: DifficultyLevel = 'intermediate'
): Promise<number> {
  const config = DIFFICULTY_LEVELS[difficulty]
  const analysis = await analyzePosition(position, difficulty)

  // Introduce random errors based on difficulty
  if (config.errorRate > 0 && Math.random() < config.errorRate) {
    const validMoves = getValidMoves(position.board)
    // Pick a random move instead of the best one
    const randomIndex = Math.floor(Math.random() * validMoves.length)
    return validMoves[randomIndex]
  }

  return analysis.bestMove
}

/**
 * Generates a comment for a move based on its score.
 */
function getMoveComment(score: number, isFirst: boolean): string {
  if (score >= EVAL_WEIGHTS.WIN) {
    return 'Winning move!'
  }
  if (score <= -EVAL_WEIGHTS.WIN) {
    return 'Loses the game'
  }
  if (isFirst) {
    return 'Best move'
  }
  if (score > 100) {
    return 'Strong alternative'
  }
  if (score > 0) {
    return 'Decent option'
  }
  if (score < -100) {
    return 'Weak move'
  }
  return 'Playable'
}

/**
 * Evaluates multiple candidate moves and ranks them.
 * Uses minimax search to score each move.
 *
 * @param position - The current game position
 * @param difficulty - The difficulty level (determines search depth)
 * @returns Promise resolving to array of moves with scores, sorted best to worst
 */
export async function rankMoves(
  position: Position,
  difficulty: DifficultyLevel = 'intermediate'
): Promise<Array<{ column: number; score: number; comment: string }>> {
  const validMoves = getValidMoves(position.board)
  const config = DIFFICULTY_LEVELS[difficulty]
  const { currentPlayer } = position

  // Evaluate each move using minimax
  const moveScores = validMoves.map((col) => {
    const result = applyMove(position.board, col, currentPlayer)
    if (!result.success || !result.board) {
      return { column: col, score: -Infinity, comment: 'Invalid move' }
    }

    // Search from opponent's perspective after this move
    const nextPlayer: Player = currentPlayer === 1 ? 2 : 1
    const { score } = minimax(
      result.board,
      config.searchDepth - 1,
      -Infinity,
      Infinity,
      false, // opponent is minimizing
      currentPlayer,
      nextPlayer
    )

    return { column: col, score, comment: '' }
  })

  // Sort by score (best first)
  moveScores.sort((a, b) => b.score - a.score)

  // Add comments
  return moveScores.map((move, index) => ({
    ...move,
    comment: getMoveComment(move.score, index === 0),
  }))
}

/**
 * Checks if the current position is part of the "solved" database.
 * Four-in-a-row is a solved game - with perfect play, the first player wins.
 *
 * STUB: Always returns false.
 * TODO: Integrate with opening book / endgame database.
 *
 * @param position - The current game position
 * @returns Whether we have perfect information about this position
 */
export function isPositionSolved(_position: Position): boolean {
  // TODO: Implement lookup in opening/endgame database
  return false
}

/**
 * Gets a human-readable summary of a position.
 *
 * @param position - The current game position
 * @returns A brief description of the game state
 */
export function describePosition(position: Position): string {
  const moveCount = position.moveHistory.length
  const validMoves = getValidMoves(position.board)

  if (validMoves.length === 0) {
    return 'Game complete'
  }

  if (moveCount === 0) {
    return 'Opening position - Player 1 to move'
  }

  if (moveCount < 7) {
    return `Early game (${moveCount} moves) - Player ${position.currentPlayer} to move`
  }

  if (moveCount < 20) {
    return `Middle game (${moveCount} moves) - Player ${position.currentPlayer} to move`
  }

  return `Late game (${moveCount} moves) - Player ${position.currentPlayer} to move`
}

/**
 * Configuration for AI difficulty levels.
 * Future implementation will use these to adjust search depth or noise.
 */
export const DIFFICULTY_LEVELS = {
  beginner: {
    name: 'Beginner',
    description: 'Makes occasional mistakes',
    searchDepth: 2,
    errorRate: 0.3,
  },
  intermediate: {
    name: 'Intermediate',
    description: 'Plays reasonably but not perfectly',
    searchDepth: 4,
    errorRate: 0.1,
  },
  expert: {
    name: 'Expert',
    description: 'Strong play, hard to beat',
    searchDepth: 8,
    errorRate: 0.02,
  },
  perfect: {
    name: 'Perfect',
    description: 'Optimal play (theoretically unbeatable)',
    searchDepth: 42, // Full game tree
    errorRate: 0,
  },
} as const

export type DifficultyLevel = keyof typeof DIFFICULTY_LEVELS
