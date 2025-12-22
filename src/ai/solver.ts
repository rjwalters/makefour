/**
 * Perfect Play Solver Module for Connect Four
 *
 * This module provides integration with a perfect play database for Connect Four.
 * Connect Four is a solved game - with perfect play, the first player always wins.
 *
 * The solver uses the connect4.gamesolver.org API by Pascal Pons, which provides
 * game-theoretic values for any position. For more information:
 * https://connect4.gamesolver.org/
 *
 * Position encoding: Moves are represented as a string of column numbers (1-7).
 * Example: "4453" means: col 4, col 4, col 5, col 3 (using 1-indexed columns)
 */

import { type Board, COLUMNS, ROWS, getValidMoves } from '../game/makefour'
import type { Position } from './coach'
import { LRUCache } from '../lib/cache'

// ============================================================================
// POSITION ENCODING
// ============================================================================

/**
 * Encodes a position's move history into the solver's format.
 * The solver uses 1-indexed columns (1-7), while our game uses 0-indexed (0-6).
 *
 * @param moveHistory - Array of column indices (0-6)
 * @returns String of column numbers (1-7) for the solver API
 */
export function encodePosition(moveHistory: number[]): string {
  return moveHistory.map((col) => col + 1).join('')
}

/**
 * Decodes a solver position string back to move history.
 *
 * @param encoded - String of column numbers (1-7)
 * @returns Array of column indices (0-6)
 */
export function decodePosition(encoded: string): number[] {
  return encoded.split('').map((c) => parseInt(c, 10) - 1)
}

// ============================================================================
// SOLVER RESULT TYPES
// ============================================================================

/**
 * Result from the solver API for a single position.
 */
export interface SolverResult {
  /** Whether the query was successful */
  success: boolean
  /** Game-theoretic score for the position (positive = win, 0 = draw, negative = loss) */
  score: number | null
  /** Array of scores for each column (index 0-6), null if column is invalid/full */
  moveScores: (number | null)[]
  /** Error message if unsuccessful */
  error?: string
  /** Whether this result came from cache */
  cached?: boolean
}

/**
 * Game-theoretic evaluation of a position.
 */
export type GameTheoreticValue = 'win' | 'loss' | 'draw' | 'unknown'

/**
 * Detailed analysis of a position from the solver.
 */
export interface SolverAnalysis {
  /** The game-theoretic value of the current position */
  value: GameTheoreticValue
  /** Score from the solver's perspective (positive = current player winning) */
  score: number
  /** Optimal move(s) - column indices (0-6) */
  optimalMoves: number[]
  /** All moves ranked by score */
  rankedMoves: Array<{ column: number; score: number; value: GameTheoreticValue }>
  /** Distance to game end (number of moves) if known */
  distanceToEnd?: number
}

// ============================================================================
// SCORE INTERPRETATION
// ============================================================================

/**
 * The solver uses a specific scoring system:
 * - Positive scores: Current player wins
 * - Zero: Draw
 * - Negative scores: Current player loses
 *
 * The magnitude indicates how many moves until the game ends:
 * - Score of N means win in (22 - N) moves
 * - Score of -N means loss in (22 - N) moves
 *
 * For a 6x7 board with 42 cells, max moves is 42.
 * Score range is typically [-21, 21] where 21 would be immediate win.
 */

const MAX_SCORE = 21 // Maximum possible score (wins immediately)

/**
 * Converts a solver score to a game-theoretic value.
 */
export function scoreToValue(score: number): GameTheoreticValue {
  if (score > 0) return 'win'
  if (score < 0) return 'loss'
  return 'draw'
}

/**
 * Calculates the distance to game end from a solver score.
 * Returns undefined if the game is drawn (infinite play possible).
 */
export function scoreToDistance(score: number): number | undefined {
  if (score === 0) return undefined
  return MAX_SCORE + 1 - Math.abs(score)
}

// ============================================================================
// SOLVER API CLIENT
// ============================================================================

const SOLVER_API_BASE = 'https://connect4.gamesolver.org/solve'

/** Maximum cache size to prevent memory issues */
const MAX_CACHE_SIZE = 10000

/** Cache for solver results to avoid redundant API calls */
const solverCache = new LRUCache<string, SolverResult>(MAX_CACHE_SIZE)

/**
 * Queries the solver API for a position.
 * Results are cached to minimize API calls.
 *
 * @param position - The game position to analyze
 * @param timeout - Timeout in milliseconds (default 5000)
 * @returns Solver result with scores for all valid moves
 */
export async function queryPosition(
  position: Position,
  timeout = 5000
): Promise<SolverResult> {
  const encoded = encodePosition(position.moveHistory)

  // Check cache first
  if (solverCache.has(encoded)) {
    return { ...solverCache.get(encoded)!, cached: true }
  }

  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    const response = await fetch(`${SOLVER_API_BASE}?pos=${encoded}`, {
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
      },
    })

    clearTimeout(timeoutId)

    if (!response.ok) {
      return {
        success: false,
        score: null,
        moveScores: Array(COLUMNS).fill(null),
        error: `API error: ${response.status}`,
      }
    }

    const data = await response.json()

    // API returns: { pos: "...", score: [...] }
    // score is an array of 7 numbers, one per column
    // 100 means the column is invalid/full
    const moveScores: (number | null)[] = (data.score as number[]).map((s: number) =>
      s === 100 ? null : s
    )

    // Calculate overall position score (best move for current player)
    const validScores = moveScores.filter((s): s is number => s !== null)
    const positionScore =
      validScores.length > 0 ? Math.max(...validScores) : 0

    const result: SolverResult = {
      success: true,
      score: positionScore,
      moveScores,
    }

    // Cache the result (LRU eviction handled automatically)
    solverCache.set(encoded, result)

    return result
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      return {
        success: false,
        score: null,
        moveScores: Array(COLUMNS).fill(null),
        error: 'Request timed out',
      }
    }
    return {
      success: false,
      score: null,
      moveScores: Array(COLUMNS).fill(null),
      error: error instanceof Error ? error.message : 'Unknown error',
    }
  }
}

/**
 * Analyzes a position using the solver and returns detailed analysis.
 *
 * @param position - The game position to analyze
 * @param timeout - Timeout in milliseconds
 * @returns Detailed solver analysis
 */
export async function analyzeWithSolver(
  position: Position,
  timeout = 5000
): Promise<SolverAnalysis | null> {
  const result = await queryPosition(position, timeout)

  if (!result.success || result.score === null) {
    return null
  }

  const validMoves = getValidMoves(position.board)

  // Build ranked moves list
  const rankedMoves: Array<{ column: number; score: number; value: GameTheoreticValue }> =
    validMoves
      .map((col) => ({
        column: col,
        score: result.moveScores[col] ?? -Infinity,
        value: result.moveScores[col] !== null ? scoreToValue(result.moveScores[col]) : ('unknown' as GameTheoreticValue),
      }))
      .filter((m) => m.score !== -Infinity)
      .sort((a, b) => b.score - a.score)

  // Find optimal moves (all moves with the best score)
  const bestScore = rankedMoves.length > 0 ? rankedMoves[0].score : 0
  const optimalMoves = rankedMoves.filter((m) => m.score === bestScore).map((m) => m.column)

  return {
    value: scoreToValue(result.score),
    score: result.score,
    optimalMoves,
    rankedMoves,
    distanceToEnd: scoreToDistance(result.score),
  }
}

// ============================================================================
// OPTIMAL MOVE SELECTION
// ============================================================================

/**
 * Gets the optimal move for a position using the perfect play solver.
 * Falls back to null if the solver is unavailable.
 *
 * @param position - The game position
 * @param timeout - Timeout in milliseconds
 * @returns The optimal column (0-6) or null if unavailable
 */
export async function getOptimalMove(
  position: Position,
  timeout = 5000
): Promise<number | null> {
  const analysis = await analyzeWithSolver(position, timeout)

  if (!analysis || analysis.optimalMoves.length === 0) {
    return null
  }

  // If multiple optimal moves exist, prefer center columns
  if (analysis.optimalMoves.length > 1) {
    const centerCol = Math.floor(COLUMNS / 2)
    const sortedByCenter = [...analysis.optimalMoves].sort(
      (a, b) => Math.abs(a - centerCol) - Math.abs(b - centerCol)
    )
    return sortedByCenter[0]
  }

  return analysis.optimalMoves[0]
}

/**
 * Checks if a position can be solved (i.e., the solver is available).
 * This is a quick check that attempts to query the solver.
 *
 * @param position - The game position
 * @param timeout - Timeout in milliseconds (default 2000 for quick check)
 * @returns Whether the position was successfully solved
 */
export async function canSolvePosition(
  position: Position,
  timeout = 2000
): Promise<boolean> {
  const result = await queryPosition(position, timeout)
  return result.success
}

// ============================================================================
// CACHE MANAGEMENT
// ============================================================================

/**
 * Clears the solver cache.
 * Useful for testing or memory management.
 */
export function clearSolverCache(): void {
  solverCache.clear()
}

/**
 * Gets the current cache size.
 */
export function getSolverCacheSize(): number {
  return solverCache.size
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Converts a board state to a move history string for debugging.
 * Note: This is a one-way operation - we can't perfectly reconstruct
 * move order from just the board state.
 */
export function boardToDebugString(board: Board): string {
  const rows: string[] = []
  for (let row = 0; row < ROWS; row++) {
    const cells = board[row].map((c) => (c === null ? '.' : c.toString())).join('')
    rows.push(cells)
  }
  return rows.join('\n')
}

/**
 * Gets a human-readable description of the solver result.
 */
export function describeSolverResult(analysis: SolverAnalysis): string {
  const { value, distanceToEnd, optimalMoves } = analysis

  let description = ''

  switch (value) {
    case 'win':
      description = distanceToEnd
        ? `Winning position - win in ${distanceToEnd} moves with perfect play`
        : 'Winning position'
      break
    case 'loss':
      description = distanceToEnd
        ? `Losing position - opponent wins in ${distanceToEnd} moves with perfect play`
        : 'Losing position'
      break
    case 'draw':
      description = 'Drawn position with perfect play'
      break
    default:
      description = 'Unknown position'
  }

  if (optimalMoves.length > 0) {
    const moveStr = optimalMoves.map((c) => c + 1).join(', ')
    description += `. Best move(s): column ${moveStr}`
  }

  return description
}
