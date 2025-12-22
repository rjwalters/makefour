/**
 * Aggressive Minimax Engine
 *
 * A minimax variant optimized for aggressive, threat-creating play.
 * Designed for the "Blitz" persona - favors creating threats over blocking,
 * prefers double-threat setups, and plays with a faster/riskier style.
 */

import type { AIEngine, EngineConfig, MoveResult } from '../ai-engine'
import { type Board, type Player, applyMove, checkWinner, COLUMNS, ROWS } from '../game'
import { getValidMoves, orderMoves, forEachWindow, countThreats } from './engine-utils'

// ============================================================================
// CONFIGURABLE EVALUATION WEIGHTS
// ============================================================================

/**
 * Evaluation weights for aggressive play style.
 * Higher threat weights, lower defense priority.
 */
export interface AggressiveEvalWeights {
  win: number
  ownThreats: number // 3-in-row with open end
  opponentThreats: number // Lower priority on blocking
  centerControl: number
  doubleThreats: number // Loves setting up forks
  twoInRow: number
}

const DEFAULT_AGGRESSIVE_WEIGHTS: AggressiveEvalWeights = {
  win: 100000,
  ownThreats: 150, // Higher than standard minimax
  opponentThreats: 80, // Lower priority on defense
  centerControl: 5,
  doubleThreats: 500, // High value for double-threat setups
  twoInRow: 15,
}

// ============================================================================
// POSITION EVALUATION (AGGRESSIVE)
// ============================================================================

/**
 * Count double threats - positions where player has 2+ threats simultaneously.
 * These are extremely valuable as opponent can only block one.
 */
function countDoubleThreats(board: Board, player: Player): number {
  const threats = countThreats(board, player)
  return threats >= 2 ? 1 : 0
}

/**
 * Evaluates a window for aggressive scoring.
 */
function evaluateWindow(
  window: (Player | null)[],
  player: Player,
  weights: AggressiveEvalWeights
): number {
  const opponent: Player = player === 1 ? 2 : 1
  const playerCount = window.filter((c) => c === player).length
  const opponentCount = window.filter((c) => c === opponent).length
  const emptyCount = window.filter((c) => c === null).length

  // Mixed windows are worthless
  if (opponentCount > 0 && playerCount > 0) return 0

  // Own pieces - prioritize threats
  if (playerCount === 4) return weights.win
  if (playerCount === 3 && emptyCount === 1) return weights.ownThreats
  if (playerCount === 2 && emptyCount === 2) return weights.twoInRow

  // Opponent pieces - lower priority on defense
  if (opponentCount === 4) return -weights.win
  if (opponentCount === 3 && emptyCount === 1) return -weights.opponentThreats
  if (opponentCount === 2 && emptyCount === 2) return -weights.twoInRow * 0.5

  return 0
}

/**
 * Aggressive position evaluation emphasizing threats and double-threats.
 */
function evaluatePositionAggressive(
  board: Board,
  player: Player,
  weights: AggressiveEvalWeights
): number {
  let score = 0
  const opponent: Player = player === 1 ? 2 : 1

  // Center column control (slightly less important for aggressive play)
  const centerCol = Math.floor(COLUMNS / 2)
  for (let row = 0; row < ROWS; row++) {
    if (board[row][centerCol] === player) {
      score += weights.centerControl
    } else if (board[row][centerCol] !== null) {
      score -= weights.centerControl
    }
  }

  // Evaluate all windows using shared iterator
  forEachWindow(board, (window) => {
    score += evaluateWindow(window, player, weights)
  })

  // Big bonus for double threats (forks)
  const ownDoubleThreats = countDoubleThreats(board, player)
  const oppDoubleThreats = countDoubleThreats(board, opponent)
  score += ownDoubleThreats * weights.doubleThreats
  score -= oppDoubleThreats * weights.doubleThreats * 0.5 // Less worried about opponent's forks

  return score
}

// ============================================================================
// MINIMAX SEARCH (AGGRESSIVE VARIANT)
// ============================================================================

interface MinimaxSearchResult {
  score: number
  move: number | null
  nodesSearched: number
}

function minimaxSearch(
  board: Board,
  depth: number,
  alpha: number,
  beta: number,
  maximizingPlayer: boolean,
  player: Player,
  currentPlayer: Player,
  deadline: number,
  nodesSearched: number,
  weights: AggressiveEvalWeights
): MinimaxSearchResult {
  nodesSearched++

  // Check time limit
  if (Date.now() > deadline) {
    return {
      score: evaluatePositionAggressive(board, player, weights),
      move: null,
      nodesSearched,
    }
  }

  // Check terminal states
  const winner = checkWinner(board)
  if (winner !== null) {
    if (winner === 'draw') return { score: 0, move: null, nodesSearched }
    const winScore = weights.win + depth * 100
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
    return {
      score: evaluatePositionAggressive(board, player, weights),
      move: null,
      nodesSearched,
    }
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
        nodesSearched,
        weights
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
        nodesSearched,
        weights
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
// AGGRESSIVE MINIMAX ENGINE
// ============================================================================

/**
 * Aggressive minimax engine for the Blitz persona.
 *
 * Characteristics:
 * - Favors creating threats over blocking
 * - High value on double-threat setups (forks)
 * - Slightly shorter search depth for faster, riskier play
 */
export class AggressiveMinimaxEngine implements AIEngine {
  readonly name = 'aggressive-minimax'
  readonly description =
    'Aggressive minimax favoring threats and double-threat setups (for Blitz)'

  private weights: AggressiveEvalWeights

  constructor(customWeights?: Partial<AggressiveEvalWeights>) {
    this.weights = { ...DEFAULT_AGGRESSIVE_WEIGHTS, ...customWeights }
  }

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

    // Get custom weights from config if provided
    const weights = config.customParams?.evalWeights
      ? { ...this.weights, ...(config.customParams.evalWeights as Partial<AggressiveEvalWeights>) }
      : this.weights

    const deadline = startTime + timeBudget * 0.9
    let bestMove = validMoves[Math.floor(validMoves.length / 2)]
    let bestScore = -Infinity
    let totalNodesSearched = 0
    let depthReached = 0

    // Iterative deepening
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
        0,
        weights
      )

      totalNodesSearched += result.nodesSearched

      if (result.move !== null) {
        bestMove = result.move
        bestScore = result.score
        depthReached = depth
      }

      // Stop early if we found a forced win
      if (Math.abs(bestScore) >= weights.win) break
    }

    // Random errors based on config
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

    const confidence = Math.min(
      1,
      Math.max(0, (bestScore + weights.win) / (2 * weights.win))
    )

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
    return evaluatePositionAggressive(board, player, this.weights)
  }

  explainMove(board: Board, move: number, player: Player): string {
    const result = applyMove(board, move, player)
    if (!result.success || !result.board) {
      return `Invalid move: column ${move}`
    }

    const winner = checkWinner(result.board)
    if (winner === player) {
      return `Winning move in column ${move}! Aggressive play pays off.`
    }

    // Check if creates double threat
    const threatsAfter = countThreats(result.board, player)
    if (threatsAfter >= 2) {
      return `Column ${move} creates a devastating double threat!`
    }

    // Check if creates a threat
    const threatsBefore = countThreats(board, player)
    if (threatsAfter > threatsBefore) {
      return `Column ${move} creates a new threat - keeping the pressure on!`
    }

    const centerCol = Math.floor(COLUMNS / 2)
    if (move === centerCol) {
      return `Taking center column ${move} for attacking position`
    }

    return `Playing column ${move} to build offensive pressure`
  }
}

// Export singleton instance with default weights
export const aggressiveMinimaxEngine = new AggressiveMinimaxEngine()
