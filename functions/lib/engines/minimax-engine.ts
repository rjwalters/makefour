/**
 * Minimax Engine
 *
 * Implements minimax with alpha-beta pruning and iterative deepening.
 * This wraps the existing bot AI implementation as a pluggable engine.
 */

import type { AIEngine, EngineConfig, MoveResult, SearchInfo } from '../ai-engine'
import {
  type Board,
  type Player,
  applyMove,
  checkWinner,
  ROWS,
  COLUMNS,
  WIN_LENGTH,
} from '../game'

// ============================================================================
// EVALUATION WEIGHTS
// ============================================================================

const EVAL_WEIGHTS = {
  WIN: 100000,
  THREE_IN_ROW: 100,
  TWO_IN_ROW: 10,
  CENTER_CONTROL: 3,
}

// ============================================================================
// POSITION EVALUATION
// ============================================================================

/**
 * Evaluates a window of 4 cells for scoring potential.
 */
function evaluateWindow(window: (Player | null)[], player: Player): number {
  const opponent: Player = player === 1 ? 2 : 1
  const playerCount = window.filter((c) => c === player).length
  const opponentCount = window.filter((c) => c === opponent).length
  const emptyCount = window.filter((c) => c === null).length

  if (opponentCount > 0 && playerCount > 0) return 0

  if (playerCount === 4) return EVAL_WEIGHTS.WIN
  if (playerCount === 3 && emptyCount === 1) return EVAL_WEIGHTS.THREE_IN_ROW
  if (playerCount === 2 && emptyCount === 2) return EVAL_WEIGHTS.TWO_IN_ROW

  if (opponentCount === 4) return -EVAL_WEIGHTS.WIN
  if (opponentCount === 3 && emptyCount === 1) return -EVAL_WEIGHTS.THREE_IN_ROW
  if (opponentCount === 2 && emptyCount === 2) return -EVAL_WEIGHTS.TWO_IN_ROW

  return 0
}

/**
 * Evaluates the board position from the perspective of the given player.
 */
function evaluatePosition(board: Board, player: Player): number {
  let score = 0

  // Center column control
  const centerCol = Math.floor(COLUMNS / 2)
  for (let row = 0; row < ROWS; row++) {
    if (board[row][centerCol] === player) {
      score += EVAL_WEIGHTS.CENTER_CONTROL
    } else if (board[row][centerCol] !== null) {
      score -= EVAL_WEIGHTS.CENTER_CONTROL
    }
  }

  // Horizontal windows
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

  // Vertical windows
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

  // Diagonal windows (down-right)
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

  // Diagonal windows (down-left)
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
// MOVE GENERATION
// ============================================================================

/**
 * Returns valid column indices for the current board.
 */
function getValidMoves(board: Board): number[] {
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
 */
function orderMoves(moves: number[]): number[] {
  const centerCol = Math.floor(COLUMNS / 2)
  return [...moves].sort((a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol))
}

// ============================================================================
// MINIMAX SEARCH
// ============================================================================

interface MinimaxSearchResult {
  score: number
  move: number | null
  nodesSearched: number
}

/**
 * Minimax search with alpha-beta pruning and time limit.
 */
function minimaxSearch(
  board: Board,
  depth: number,
  alpha: number,
  beta: number,
  maximizingPlayer: boolean,
  player: Player,
  currentPlayer: Player,
  deadline: number,
  nodesSearched: number
): MinimaxSearchResult {
  nodesSearched++

  // Check time limit
  if (Date.now() > deadline) {
    return { score: evaluatePosition(board, player), move: null, nodesSearched }
  }

  // Check terminal states
  const winner = checkWinner(board)
  if (winner !== null) {
    if (winner === 'draw') return { score: 0, move: null, nodesSearched }
    const winScore = EVAL_WEIGHTS.WIN + depth * 100
    return {
      score: winner === player ? winScore : -winScore,
      move: null,
      nodesSearched,
    }
  }

  const validMoves = getValidMoves(board)
  if (validMoves.length === 0) return { score: 0, move: null, nodesSearched }

  // Depth limit reached
  if (depth === 0) {
    return { score: evaluatePosition(board, player), move: null, nodesSearched }
  }

  const orderedMoves = orderMoves(validMoves)
  const nextPlayer: Player = currentPlayer === 1 ? 2 : 1

  if (maximizingPlayer) {
    let maxScore = -Infinity
    let bestMove = orderedMoves[0]

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const searchResult = minimaxSearch(
        result.board,
        depth - 1,
        alpha,
        beta,
        false,
        player,
        nextPlayer,
        deadline,
        nodesSearched
      )
      nodesSearched = searchResult.nodesSearched

      if (searchResult.score > maxScore) {
        maxScore = searchResult.score
        bestMove = move
      }

      alpha = Math.max(alpha, searchResult.score)
      if (beta <= alpha) break
    }

    return { score: maxScore, move: bestMove, nodesSearched }
  } else {
    let minScore = Infinity
    let bestMove = orderedMoves[0]

    for (const move of orderedMoves) {
      const result = applyMove(board, move, currentPlayer)
      if (!result.success || !result.board) continue

      const searchResult = minimaxSearch(
        result.board,
        depth - 1,
        alpha,
        beta,
        true,
        player,
        nextPlayer,
        deadline,
        nodesSearched
      )
      nodesSearched = searchResult.nodesSearched

      if (searchResult.score < minScore) {
        minScore = searchResult.score
        bestMove = move
      }

      beta = Math.min(beta, searchResult.score)
      if (beta <= alpha) break
    }

    return { score: minScore, move: bestMove, nodesSearched }
  }
}

// ============================================================================
// MINIMAX ENGINE
// ============================================================================

/**
 * Minimax engine with alpha-beta pruning and iterative deepening.
 */
export class MinimaxEngine implements AIEngine {
  readonly name = 'minimax'
  readonly description = 'Minimax search with alpha-beta pruning and iterative deepening'

  async selectMove(
    board: Board,
    player: Player,
    config: EngineConfig,
    timeBudget: number
  ): Promise<MoveResult> {
    const startTime = Date.now()
    const validMoves = getValidMoves(board)

    if (validMoves.length === 0) {
      throw new Error('No valid moves available')
    }

    // If only one move, return it immediately
    if (validMoves.length === 1) {
      return {
        column: validMoves[0],
        confidence: 1,
        searchInfo: {
          depth: 0,
          nodesSearched: 1,
          timeUsed: Date.now() - startTime,
        },
      }
    }

    const deadline = startTime + timeBudget * 0.9 // Leave 10% buffer
    let bestMove = validMoves[Math.floor(validMoves.length / 2)] // Start with center-ish
    let bestScore = -Infinity
    let totalNodesSearched = 0
    let depthReached = 0

    // Iterative deepening: search progressively deeper until time runs out
    for (let depth = 1; depth <= config.searchDepth; depth++) {
      if (Date.now() > deadline) break

      const result = minimaxSearch(
        board,
        depth,
        -Infinity,
        Infinity,
        true,
        player,
        player,
        deadline,
        0
      )

      totalNodesSearched += result.nodesSearched

      if (result.move !== null) {
        bestMove = result.move
        bestScore = result.score
        depthReached = depth
      }

      // Stop early if we found a forced win
      if (Math.abs(bestScore) >= EVAL_WEIGHTS.WIN) break
    }

    // Introduce random errors based on config
    if (config.errorRate > 0 && Math.random() < config.errorRate) {
      const randomIndex = Math.floor(Math.random() * validMoves.length)
      return {
        column: validMoves[randomIndex],
        confidence: 0,
        searchInfo: {
          depth: depthReached,
          nodesSearched: totalNodesSearched,
          timeUsed: Date.now() - startTime,
        },
      }
    }

    // Calculate confidence based on score margin
    const confidence = Math.min(1, Math.max(0, (bestScore + EVAL_WEIGHTS.WIN) / (2 * EVAL_WEIGHTS.WIN)))

    return {
      column: bestMove,
      confidence,
      searchInfo: {
        depth: depthReached,
        nodesSearched: totalNodesSearched,
        timeUsed: Date.now() - startTime,
      },
    }
  }

  evaluatePosition(board: Board, player: Player): number {
    return evaluatePosition(board, player)
  }

  explainMove(board: Board, move: number, player: Player): string {
    const result = applyMove(board, move, player)
    if (!result.success || !result.board) {
      return `Invalid move: column ${move}`
    }

    const winner = checkWinner(result.board)
    if (winner === player) {
      return `Winning move in column ${move}!`
    }

    // Check if it blocks opponent's win
    const opponent: Player = player === 1 ? 2 : 1
    const opponentResult = applyMove(board, move, opponent)
    if (opponentResult.success && opponentResult.board) {
      const opponentWin = checkWinner(opponentResult.board)
      if (opponentWin === opponent) {
        return `Blocking opponent's winning move in column ${move}`
      }
    }

    // Check center control
    const centerCol = Math.floor(COLUMNS / 2)
    if (move === centerCol) {
      return `Taking center column ${move} for positional advantage`
    }

    return `Playing column ${move} to improve position`
  }
}

// Export singleton instance
export const minimaxEngine = new MinimaxEngine()
