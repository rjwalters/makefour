/**
 * Move Quality Scoring Module
 *
 * Scores each move as optimal/good/neutral/mistake/blunder by comparing
 * player moves against perfect play.
 */

import { type Board, type Player, getStateAtMove } from '../game/makefour'
import { rankMoves, type Position, EVAL_WEIGHTS } from './coach'

/**
 * Move quality tiers based on comparison to optimal play.
 */
export type MoveQuality = 'optimal' | 'good' | 'neutral' | 'mistake' | 'blunder'

/**
 * Analysis result for a single move.
 */
export interface MoveAnalysis {
  /** The move index (0-based) */
  moveIndex: number
  /** The column that was played */
  column: number
  /** The player who made this move */
  player: Player
  /** Quality classification of the move */
  quality: MoveQuality
  /** The optimal move for this position */
  optimalMove: number
  /** Score of the move played */
  moveScore: number
  /** Score of the optimal move */
  optimalScore: number
  /** Score difference (negative means worse than optimal) */
  scoreDifference: number
}

/**
 * Summary statistics for a game's move quality.
 */
export interface GameQualitySummary {
  /** Total number of moves */
  totalMoves: number
  /** Count of each quality tier */
  counts: Record<MoveQuality, number>
  /** Percentage of optimal moves */
  optimalPercentage: number
  /** Percentage of good or better moves */
  goodOrBetterPercentage: number
  /** Number of blunders */
  blunderCount: number
  /** Number of mistakes */
  mistakeCount: number
  /** Per-player statistics */
  byPlayer: {
    1: { moves: number; optimal: number; blunders: number }
    2: { moves: number; optimal: number; blunders: number }
  }
}

/**
 * Score thresholds for classifying move quality.
 * Based on the evaluation weights in coach.ts:
 * - WIN: 100,000
 * - THREE_IN_ROW: 100
 * - TWO_IN_ROW: 10
 */
const QUALITY_THRESHOLDS = {
  /** Moves within this score of optimal are considered "good" */
  GOOD: 50,
  /** Moves within this score of optimal are considered "neutral" */
  NEUTRAL: 200,
  /** Moves worse than this are "blunders" (losing winning position or missing forced win) */
  BLUNDER: EVAL_WEIGHTS.WIN / 2, // 50,000
}

/**
 * Classifies move quality based on score difference from optimal.
 *
 * @param moveScore - Score of the move played
 * @param optimalScore - Score of the optimal move
 * @param wasWinning - Whether the position was winning before this move
 * @param isNowLosing - Whether the position is now losing after this move
 * @returns Quality classification
 */
export function classifyMoveQuality(
  moveScore: number,
  optimalScore: number,
  wasWinning: boolean,
  isNowLosing: boolean
): MoveQuality {
  const scoreDiff = optimalScore - moveScore

  // Optimal: best possible move (or within negligible margin)
  if (scoreDiff <= 0) {
    return 'optimal'
  }

  // Blunder: turns winning position to losing, or misses forced win
  if (wasWinning && isNowLosing) {
    return 'blunder'
  }

  // Blunder: massive score drop
  if (scoreDiff >= QUALITY_THRESHOLDS.BLUNDER) {
    return 'blunder'
  }

  // Good: maintains position or small inaccuracy
  if (scoreDiff <= QUALITY_THRESHOLDS.GOOD) {
    return 'good'
  }

  // Neutral: moderate inaccuracy but doesn't change evaluation significantly
  if (scoreDiff <= QUALITY_THRESHOLDS.NEUTRAL) {
    return 'neutral'
  }

  // Mistake: significant loss of advantage
  return 'mistake'
}

/**
 * Analyzes a single move's quality by comparing to optimal play.
 *
 * @param board - Board state BEFORE the move
 * @param currentPlayer - Player making the move
 * @param playedColumn - The column that was played
 * @param moveIndex - Index of this move in the game
 * @param moveHistory - Move history up to (but not including) this move
 * @returns Move analysis result
 */
export async function analyzeSingleMove(
  board: Board,
  currentPlayer: Player,
  playedColumn: number,
  moveIndex: number,
  moveHistory: number[]
): Promise<MoveAnalysis> {
  const position: Position = {
    board,
    currentPlayer,
    moveHistory,
  }

  // Get ranked moves using expert difficulty for good accuracy with reasonable performance
  // (depth 8 vs depth 42 for 'perfect' - much faster while still accurate for most positions)
  const rankedMoves = await rankMoves(position, 'expert')

  // Find the played move in ranked moves
  const playedMoveData = rankedMoves.find((m) => m.column === playedColumn)
  const optimalMoveData = rankedMoves[0]

  const moveScore = playedMoveData?.score ?? -Infinity
  const optimalScore = optimalMoveData?.score ?? 0
  const optimalMove = optimalMoveData?.column ?? playedColumn

  // Determine if position was winning and if it's now losing
  const wasWinning = optimalScore >= EVAL_WEIGHTS.WIN
  const isNowLosing = moveScore <= -EVAL_WEIGHTS.WIN

  const quality = classifyMoveQuality(moveScore, optimalScore, wasWinning, isNowLosing)

  return {
    moveIndex,
    column: playedColumn,
    player: currentPlayer,
    quality,
    optimalMove,
    moveScore,
    optimalScore,
    scoreDifference: moveScore - optimalScore,
  }
}

/**
 * Analyzes all moves in a game for quality.
 *
 * @param moves - Array of column indices representing each move
 * @returns Promise resolving to array of move analyses
 */
export async function analyzeGame(moves: number[]): Promise<MoveAnalysis[]> {
  const analyses: MoveAnalysis[] = []

  for (let i = 0; i < moves.length; i++) {
    // Get game state BEFORE this move
    const stateBefore = getStateAtMove(moves, i)
    if (!stateBefore) continue

    const analysis = await analyzeSingleMove(
      stateBefore.board,
      stateBefore.currentPlayer,
      moves[i],
      i,
      moves.slice(0, i)
    )

    analyses.push(analysis)
  }

  return analyses
}

/**
 * Generates summary statistics for a game's move quality.
 *
 * @param analyses - Array of move analyses
 * @returns Summary statistics
 */
export function summarizeGameQuality(analyses: MoveAnalysis[]): GameQualitySummary {
  const counts: Record<MoveQuality, number> = {
    optimal: 0,
    good: 0,
    neutral: 0,
    mistake: 0,
    blunder: 0,
  }

  const byPlayer = {
    1: { moves: 0, optimal: 0, blunders: 0 },
    2: { moves: 0, optimal: 0, blunders: 0 },
  }

  for (const analysis of analyses) {
    counts[analysis.quality]++
    byPlayer[analysis.player].moves++

    if (analysis.quality === 'optimal') {
      byPlayer[analysis.player].optimal++
    }
    if (analysis.quality === 'blunder') {
      byPlayer[analysis.player].blunders++
    }
  }

  const totalMoves = analyses.length
  const optimalPercentage = totalMoves > 0 ? (counts.optimal / totalMoves) * 100 : 0
  const goodOrBetterPercentage =
    totalMoves > 0 ? ((counts.optimal + counts.good) / totalMoves) * 100 : 0

  return {
    totalMoves,
    counts,
    optimalPercentage,
    goodOrBetterPercentage,
    blunderCount: counts.blunder,
    mistakeCount: counts.mistake,
    byPlayer,
  }
}

/**
 * Gets the CSS color class for a move quality tier.
 *
 * @param quality - The move quality
 * @returns Tailwind CSS class for the color
 */
export function getMoveQualityColor(quality: MoveQuality): string {
  switch (quality) {
    case 'optimal':
      return 'bg-green-500 text-white'
    case 'good':
      return 'bg-green-300 text-green-900 dark:bg-green-700 dark:text-green-100'
    case 'neutral':
      return 'bg-yellow-300 text-yellow-900 dark:bg-yellow-700 dark:text-yellow-100'
    case 'mistake':
      return 'bg-orange-400 text-orange-900 dark:bg-orange-600 dark:text-orange-100'
    case 'blunder':
      return 'bg-red-500 text-white'
  }
}

/**
 * Gets a human-readable label for move quality.
 */
export function getMoveQualityLabel(quality: MoveQuality): string {
  switch (quality) {
    case 'optimal':
      return 'Optimal'
    case 'good':
      return 'Good'
    case 'neutral':
      return 'OK'
    case 'mistake':
      return 'Mistake'
    case 'blunder':
      return 'Blunder'
  }
}
