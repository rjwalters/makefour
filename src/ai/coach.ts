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
import {
  getOptimalMove,
  analyzeWithSolver,
  canSolvePosition,
  describeSolverResult,
  type SolverAnalysis,
} from './solver'

// ============================================================================
// POSITION EVALUATION
// ============================================================================

/**
 * Weights for position evaluation heuristics.
 */
export const EVAL_WEIGHTS = {
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
 * For "perfect" difficulty, uses the perfect play solver API.
 * For other difficulties, uses minimax search with alpha-beta pruning.
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

  // For perfect difficulty, try to use the solver first
  if (difficulty === 'perfect') {
    const optimalMove = await getOptimalMove(position)
    if (optimalMove !== null) {
      return optimalMove
    }
    // Fall back to deep minimax if solver unavailable
  }

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
 * Checks if the current position can be solved using the perfect play database.
 * Connect Four is a solved game - with perfect play, the first player always wins.
 *
 * This function queries the solver API to determine if we have perfect
 * information about this position.
 *
 * @param position - The current game position
 * @param timeout - Timeout in milliseconds for the solver query
 * @returns Promise resolving to whether we have perfect information about this position
 */
export async function isPositionSolved(
  position: Position,
  timeout = 2000
): Promise<boolean> {
  return canSolvePosition(position, timeout)
}

/**
 * Synchronous version of isPositionSolved for backwards compatibility.
 * Always returns false - use the async version for actual solver access.
 *
 * @deprecated Use the async isPositionSolved instead
 * @param _position - The current game position
 * @returns Always false (use async version for real checks)
 */
export function isPositionSolvedSync(_position: Position): boolean {
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

// ============================================================================
// THREAT DETECTION
// ============================================================================

/**
 * Represents a threat on the board.
 */
export interface Threat {
  /** Column that completes the threat (0-6) */
  column: number
  /** Row where the piece would land (0-5) */
  row: number
  /** Player who benefits from this threat */
  player: Player
  /** Type of threat */
  type: 'win' | 'block'
}

/**
 * Result of threat analysis.
 */
export interface ThreatAnalysis {
  /** Columns where current player can win immediately */
  winningMoves: number[]
  /** Columns where opponent would win (must block) */
  blockingMoves: number[]
  /** All threats on the board */
  threats: Threat[]
}

/**
 * Gets the row where a piece would land in a column.
 * @param board - Current board state
 * @param column - Column to check
 * @returns Row index (0-5) or -1 if column is full
 */
function getDropRow(board: Board, column: number): number {
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row][column] === null) {
      return row
    }
  }
  return -1
}

/**
 * Checks if playing in a column would result in a win for the given player.
 * @param board - Current board state
 * @param column - Column to check
 * @param player - Player to check for
 * @returns true if playing here wins
 */
function wouldWin(board: Board, column: number, player: Player): boolean {
  const row = getDropRow(board, column)
  if (row === -1) return false

  const result = applyMove(board, column, player)
  if (!result.success || !result.board) return false

  const winner = checkWinner(result.board)
  return winner === player
}

/**
 * Analyzes the current position for threats.
 * Identifies winning moves for current player and blocking moves needed.
 *
 * @param board - Current board state
 * @param currentPlayer - The player whose turn it is
 * @returns ThreatAnalysis with winning and blocking moves
 */
export function analyzeThreats(board: Board, currentPlayer: Player): ThreatAnalysis {
  const opponent: Player = currentPlayer === 1 ? 2 : 1
  const winningMoves: number[] = []
  const blockingMoves: number[] = []
  const threats: Threat[] = []

  for (let col = 0; col < COLUMNS; col++) {
    const row = getDropRow(board, col)
    if (row === -1) continue

    // Check if current player can win
    if (wouldWin(board, col, currentPlayer)) {
      winningMoves.push(col)
      threats.push({ column: col, row, player: currentPlayer, type: 'win' })
    }

    // Check if opponent would win (need to block)
    if (wouldWin(board, col, opponent)) {
      blockingMoves.push(col)
      threats.push({ column: col, row, player: opponent, type: 'block' })
    }
  }

  return { winningMoves, blockingMoves, threats }
}

/**
 * Gets a quick position evaluation for display purposes.
 * Uses a shallow search for speed.
 *
 * @param board - Current board state
 * @param currentPlayer - The player whose turn it is
 * @returns Evaluation score and description
 */
export function getQuickEvaluation(
  board: Board,
  currentPlayer: Player
): { score: number; description: string; result: 'win' | 'loss' | 'draw' | 'unknown' } {
  // Check for immediate wins/losses
  const threats = analyzeThreats(board, currentPlayer)

  if (threats.winningMoves.length > 0) {
    return {
      score: EVAL_WEIGHTS.WIN,
      description: 'Winning position - can win this move',
      result: 'win',
    }
  }

  // If opponent has multiple threats and we can only block one, we're losing
  if (threats.blockingMoves.length > 1) {
    return {
      score: -EVAL_WEIGHTS.WIN,
      description: 'Losing position - opponent has multiple threats',
      result: 'loss',
    }
  }

  // Use shallow evaluation for quick feedback
  const score = evaluatePosition(board, currentPlayer)

  return {
    score,
    description: getEvaluationDescription(score),
    result: getTheoreticalResult(score),
  }
}

// ============================================================================
// PERFECT PLAY ANALYSIS
// ============================================================================

/**
 * Extended analysis result that includes perfect play solver data.
 */
export interface PerfectAnalysis extends Analysis {
  /** Solver analysis if available */
  solverAnalysis: SolverAnalysis | null
  /** Whether solver data was used */
  usedSolver: boolean
}

/**
 * Analyzes a position using the perfect play solver when available.
 * Falls back to minimax analysis if the solver is unavailable.
 *
 * This provides the most accurate analysis possible by combining:
 * - Perfect play solver data (when available)
 * - Minimax heuristic analysis (as fallback or supplement)
 *
 * @param position - The game position to analyze
 * @param timeout - Timeout for solver query in milliseconds
 * @returns Promise resolving to comprehensive analysis
 */
export async function analyzeWithPerfectPlay(
  position: Position,
  timeout = 5000
): Promise<PerfectAnalysis> {
  // Try solver first
  const solverAnalysis = await analyzeWithSolver(position, timeout)

  if (solverAnalysis) {
    // We have perfect information
    const optimalMove = solverAnalysis.optimalMoves[0] ?? getValidMoves(position.board)[0] ?? 0

    return {
      bestMove: optimalMove,
      score: solverAnalysis.score * 1000, // Scale to match minimax scores
      evaluation: describeSolverResult(solverAnalysis),
      theoreticalResult: solverAnalysis.value === 'unknown' ? 'unknown' : solverAnalysis.value,
      confidence: 1, // Perfect information
      solverAnalysis,
      usedSolver: true,
    }
  }

  // Fall back to minimax analysis
  const analysis = await analyzePosition(position, 'expert')

  return {
    ...analysis,
    solverAnalysis: null,
    usedSolver: false,
  }
}

/**
 * Gets all optimal moves for a position using the perfect play solver.
 * Returns all moves that maintain the optimal game-theoretic result.
 *
 * @param position - The game position
 * @param timeout - Timeout in milliseconds
 * @returns Array of optimal column indices (0-6), or null if solver unavailable
 */
export async function getOptimalMoves(
  position: Position,
  timeout = 5000
): Promise<number[] | null> {
  const analysis = await analyzeWithSolver(position, timeout)
  return analysis?.optimalMoves ?? null
}

/**
 * Scores a move against perfect play.
 * Returns a rating of how good the move is compared to the optimal move.
 *
 * @param position - The game position before the move
 * @param column - The column that was played (0-6)
 * @param timeout - Timeout in milliseconds
 * @returns Object with rating and explanation, or null if solver unavailable
 */
export async function scoreMove(
  position: Position,
  column: number,
  timeout = 5000
): Promise<{ rating: 'optimal' | 'good' | 'inaccuracy' | 'mistake' | 'blunder'; explanation: string } | null> {
  const analysis = await analyzeWithSolver(position, timeout)

  if (!analysis) {
    return null
  }

  const moveData = analysis.rankedMoves.find((m) => m.column === column)
  if (!moveData) {
    return { rating: 'blunder', explanation: 'Invalid move' }
  }

  const isOptimal = analysis.optimalMoves.includes(column)
  const bestScore = analysis.rankedMoves[0]?.score ?? 0
  const scoreDiff = bestScore - moveData.score

  if (isOptimal) {
    return {
      rating: 'optimal',
      explanation: `Perfect move! This is one of the ${analysis.optimalMoves.length} optimal move(s).`,
    }
  }

  // Categorize the move based on how much worse it is than optimal
  if (scoreDiff <= 2) {
    return {
      rating: 'good',
      explanation: 'Good move, very close to optimal play.',
    }
  }

  if (scoreDiff <= 5) {
    return {
      rating: 'inaccuracy',
      explanation: 'Slight inaccuracy. A better move was available.',
    }
  }

  // Check if this move changes the game-theoretic result
  const optimalValue = analysis.value
  const moveValue = moveData.value

  if (optimalValue === 'win' && moveValue === 'draw') {
    return {
      rating: 'mistake',
      explanation: 'Mistake! This move turns a winning position into a draw.',
    }
  }

  if (optimalValue === 'win' && moveValue === 'loss') {
    return {
      rating: 'blunder',
      explanation: 'Blunder! This move turns a winning position into a loss.',
    }
  }

  if (optimalValue === 'draw' && moveValue === 'loss') {
    return {
      rating: 'blunder',
      explanation: 'Blunder! This move turns a drawn position into a loss.',
    }
  }

  if (scoreDiff <= 10) {
    return {
      rating: 'mistake',
      explanation: 'Mistake. There was a significantly better move.',
    }
  }

  return {
    rating: 'blunder',
    explanation: 'Blunder! This move seriously weakens your position.',
  }
}

// Re-export solver utilities for convenience
export { encodePosition, decodePosition, clearSolverCache, getSolverCacheSize } from './solver'
