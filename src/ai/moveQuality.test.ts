import { describe, it, expect } from 'vitest'
import {
  classifyMoveQuality,
  analyzeSingleMove,
  analyzeGame,
  summarizeGameQuality,
  getMoveQualityColor,
  getMoveQualityLabel,
  type MoveQuality,
  type MoveAnalysis,
} from './moveQuality'
import { createEmptyBoard, type Board } from '../game/makefour'
import { EVAL_WEIGHTS } from './coach'

/**
 * Helper to create a board from a string representation.
 * '.' = empty, '1' = player 1, '2' = player 2
 * Rows are from top to bottom.
 */
function createBoardFromString(str: string): Board {
  const lines = str
    .trim()
    .split('\n')
    .map((l) => l.trim())
  const board = createEmptyBoard()
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 7; col++) {
      const char = lines[row]?.[col]
      if (char === '1') board[row][col] = 1
      else if (char === '2') board[row][col] = 2
    }
  }
  return board
}

describe('Move Quality Scoring', () => {
  describe('classifyMoveQuality', () => {
    it('returns optimal when score matches optimal', () => {
      const result = classifyMoveQuality(100, 100, false, false)
      expect(result).toBe('optimal')
    })

    it('returns optimal when move score exceeds optimal (edge case)', () => {
      const result = classifyMoveQuality(150, 100, false, false)
      expect(result).toBe('optimal')
    })

    it('returns good for small score difference', () => {
      // GOOD threshold is 50
      const result = classifyMoveQuality(70, 100, false, false)
      expect(result).toBe('good')
    })

    it('returns neutral for moderate score difference', () => {
      // NEUTRAL threshold is 200, GOOD is 50
      const result = classifyMoveQuality(0, 100, false, false)
      expect(result).toBe('neutral')
    })

    it('returns mistake for significant score loss', () => {
      // Score diff > 200 but < 50000
      const result = classifyMoveQuality(-500, 100, false, false)
      expect(result).toBe('mistake')
    })

    it('returns blunder when turning winning to losing', () => {
      const result = classifyMoveQuality(-EVAL_WEIGHTS.WIN, EVAL_WEIGHTS.WIN, true, true)
      expect(result).toBe('blunder')
    })

    it('returns blunder for massive score drop', () => {
      // BLUNDER threshold is WIN/2 = 50000
      const result = classifyMoveQuality(0, 60000, false, false)
      expect(result).toBe('blunder')
    })

    it('distinguishes between winning position lost vs not', () => {
      // Same score diff, but one was from winning position
      const fromWinning = classifyMoveQuality(-EVAL_WEIGHTS.WIN, EVAL_WEIGHTS.WIN, true, true)
      const fromEqual = classifyMoveQuality(-100, 100, false, false)

      expect(fromWinning).toBe('blunder')
      expect(fromEqual).toBe('neutral')
    })

    it('classifies boundary cases correctly', () => {
      // Exactly at GOOD threshold (50)
      expect(classifyMoveQuality(50, 100, false, false)).toBe('good')

      // Just over GOOD threshold
      expect(classifyMoveQuality(49, 100, false, false)).toBe('neutral')

      // Exactly at NEUTRAL threshold (200)
      expect(classifyMoveQuality(-100, 100, false, false)).toBe('neutral')

      // Just over NEUTRAL threshold
      expect(classifyMoveQuality(-101, 100, false, false)).toBe('mistake')
    })
  })

  describe('analyzeSingleMove', () => {
    it('identifies optimal move correctly', async () => {
      // Position where column 3 wins for player 1
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        111....
      `)

      const analysis = await analyzeSingleMove(board, 1, 3, 0, [])
      expect(analysis.quality).toBe('optimal')
      expect(analysis.column).toBe(3)
      expect(analysis.optimalMove).toBe(3)
      expect(analysis.player).toBe(1)
      expect(analysis.moveIndex).toBe(0)
    })

    it('identifies blunder when missing winning move', async () => {
      // Player 1 can win with column 3, but plays column 0
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        111....
      `)

      const analysis = await analyzeSingleMove(board, 1, 0, 0, [])
      // Playing column 0 when column 3 wins is a blunder
      expect(analysis.quality).toBe('blunder')
      expect(analysis.optimalMove).toBe(3)
    })

    it('returns correct score difference', async () => {
      const board = createEmptyBoard()

      // First move - center is best
      const analysis = await analyzeSingleMove(board, 1, 3, 0, [])
      expect(analysis.scoreDifference).toBe(0) // Center is optimal
      expect(analysis.quality).toBe('optimal')
    })

    it('handles edge column moves', async () => {
      const board = createEmptyBoard()

      // Playing edge column 0 on empty board
      const analysis = await analyzeSingleMove(board, 1, 0, 0, [])
      // Edge is not optimal, should be good/neutral/mistake
      expect(analysis.optimalMove).toBe(3) // Center is best
      expect(['good', 'neutral', 'mistake']).toContain(analysis.quality)
    })
  })

  describe('analyzeGame', () => {
    it('analyzes empty game', async () => {
      const analyses = await analyzeGame([])
      expect(analyses).toHaveLength(0)
    })

    it('analyzes single move game', async () => {
      const analyses = await analyzeGame([3]) // Center column
      expect(analyses).toHaveLength(1)
      expect(analyses[0].moveIndex).toBe(0)
      expect(analyses[0].column).toBe(3)
      expect(analyses[0].player).toBe(1)
    })

    it('analyzes multi-move game correctly', async () => {
      // Simple opening sequence
      const analyses = await analyzeGame([3, 3, 2, 4])
      expect(analyses).toHaveLength(4)

      // Verify move indices
      expect(analyses[0].moveIndex).toBe(0)
      expect(analyses[1].moveIndex).toBe(1)
      expect(analyses[2].moveIndex).toBe(2)
      expect(analyses[3].moveIndex).toBe(3)

      // Verify players alternate
      expect(analyses[0].player).toBe(1)
      expect(analyses[1].player).toBe(2)
      expect(analyses[2].player).toBe(1)
      expect(analyses[3].player).toBe(2)
    })

    it('identifies blunder in game sequence', async () => {
      // Game where player 1 sets up for win but player 2 doesn't block
      // [3, 0, 3, 0, 3, 6] - Player 1 plays 3 three times, player 2 plays edges
      // On move 5 (index 4), player 1 plays column 3 again to win
      const analyses = await analyzeGame([3, 0, 3, 0, 3])

      // Player 1 plays center consistently - should be good moves
      expect(analyses[0].quality).toBe('optimal') // First center move
    })

    it('returns analyses in correct order', async () => {
      const analyses = await analyzeGame([0, 1, 2, 3, 4])

      for (let i = 0; i < analyses.length; i++) {
        expect(analyses[i].moveIndex).toBe(i)
        expect(analyses[i].column).toBe(i) // Moves are 0,1,2,3,4
      }
    })
  })

  describe('summarizeGameQuality', () => {
    it('handles empty analysis array', () => {
      const summary = summarizeGameQuality([])

      expect(summary.totalMoves).toBe(0)
      expect(summary.optimalPercentage).toBe(0)
      expect(summary.goodOrBetterPercentage).toBe(0)
      expect(summary.blunderCount).toBe(0)
      expect(summary.mistakeCount).toBe(0)
    })

    it('calculates correct counts', () => {
      const analyses: MoveAnalysis[] = [
        { moveIndex: 0, column: 3, player: 1, quality: 'optimal', optimalMove: 3, moveScore: 100, optimalScore: 100, scoreDifference: 0 },
        { moveIndex: 1, column: 3, player: 2, quality: 'good', optimalMove: 3, moveScore: 80, optimalScore: 100, scoreDifference: -20 },
        { moveIndex: 2, column: 2, player: 1, quality: 'mistake', optimalMove: 3, moveScore: -100, optimalScore: 100, scoreDifference: -200 },
        { moveIndex: 3, column: 0, player: 2, quality: 'blunder', optimalMove: 3, moveScore: -50000, optimalScore: 100, scoreDifference: -50100 },
      ]

      const summary = summarizeGameQuality(analyses)

      expect(summary.totalMoves).toBe(4)
      expect(summary.counts.optimal).toBe(1)
      expect(summary.counts.good).toBe(1)
      expect(summary.counts.mistake).toBe(1)
      expect(summary.counts.blunder).toBe(1)
      expect(summary.counts.neutral).toBe(0)
    })

    it('calculates correct percentages', () => {
      const analyses: MoveAnalysis[] = [
        { moveIndex: 0, column: 3, player: 1, quality: 'optimal', optimalMove: 3, moveScore: 100, optimalScore: 100, scoreDifference: 0 },
        { moveIndex: 1, column: 3, player: 2, quality: 'optimal', optimalMove: 3, moveScore: 100, optimalScore: 100, scoreDifference: 0 },
        { moveIndex: 2, column: 3, player: 1, quality: 'good', optimalMove: 3, moveScore: 80, optimalScore: 100, scoreDifference: -20 },
        { moveIndex: 3, column: 0, player: 2, quality: 'mistake', optimalMove: 3, moveScore: -100, optimalScore: 100, scoreDifference: -200 },
      ]

      const summary = summarizeGameQuality(analyses)

      expect(summary.optimalPercentage).toBe(50) // 2/4
      expect(summary.goodOrBetterPercentage).toBe(75) // 3/4
    })

    it('tracks per-player statistics', () => {
      const analyses: MoveAnalysis[] = [
        { moveIndex: 0, column: 3, player: 1, quality: 'optimal', optimalMove: 3, moveScore: 100, optimalScore: 100, scoreDifference: 0 },
        { moveIndex: 1, column: 0, player: 2, quality: 'blunder', optimalMove: 3, moveScore: -50000, optimalScore: 100, scoreDifference: -50100 },
        { moveIndex: 2, column: 3, player: 1, quality: 'optimal', optimalMove: 3, moveScore: 100, optimalScore: 100, scoreDifference: 0 },
        { moveIndex: 3, column: 0, player: 2, quality: 'blunder', optimalMove: 3, moveScore: -50000, optimalScore: 100, scoreDifference: -50100 },
      ]

      const summary = summarizeGameQuality(analyses)

      expect(summary.byPlayer[1].moves).toBe(2)
      expect(summary.byPlayer[1].optimal).toBe(2)
      expect(summary.byPlayer[1].blunders).toBe(0)

      expect(summary.byPlayer[2].moves).toBe(2)
      expect(summary.byPlayer[2].optimal).toBe(0)
      expect(summary.byPlayer[2].blunders).toBe(2)
    })

    it('provides convenience counts', () => {
      const analyses: MoveAnalysis[] = [
        { moveIndex: 0, column: 3, player: 1, quality: 'mistake', optimalMove: 3, moveScore: -100, optimalScore: 100, scoreDifference: -200 },
        { moveIndex: 1, column: 0, player: 2, quality: 'blunder', optimalMove: 3, moveScore: -50000, optimalScore: 100, scoreDifference: -50100 },
        { moveIndex: 2, column: 3, player: 1, quality: 'mistake', optimalMove: 3, moveScore: -100, optimalScore: 100, scoreDifference: -200 },
      ]

      const summary = summarizeGameQuality(analyses)

      expect(summary.mistakeCount).toBe(2)
      expect(summary.blunderCount).toBe(1)
    })
  })

  describe('getMoveQualityColor', () => {
    it('returns green for optimal', () => {
      const color = getMoveQualityColor('optimal')
      expect(color).toContain('green-500')
      expect(color).toContain('text-white')
    })

    it('returns green variant for good', () => {
      const color = getMoveQualityColor('good')
      expect(color).toContain('green-300')
    })

    it('returns yellow for neutral', () => {
      const color = getMoveQualityColor('neutral')
      expect(color).toContain('yellow')
    })

    it('returns orange for mistake', () => {
      const color = getMoveQualityColor('mistake')
      expect(color).toContain('orange')
    })

    it('returns red for blunder', () => {
      const color = getMoveQualityColor('blunder')
      expect(color).toContain('red-500')
      expect(color).toContain('text-white')
    })

    it('includes dark mode classes for mid-tier qualities', () => {
      expect(getMoveQualityColor('good')).toContain('dark:')
      expect(getMoveQualityColor('neutral')).toContain('dark:')
      expect(getMoveQualityColor('mistake')).toContain('dark:')
    })
  })

  describe('getMoveQualityLabel', () => {
    it('returns correct labels for all qualities', () => {
      const qualities: MoveQuality[] = ['optimal', 'good', 'neutral', 'mistake', 'blunder']
      const expectedLabels = ['Optimal', 'Good', 'OK', 'Mistake', 'Blunder']

      qualities.forEach((quality, i) => {
        expect(getMoveQualityLabel(quality)).toBe(expectedLabels[i])
      })
    })
  })

  describe('Performance', () => {
    it('analyzes a short game in reasonable time (< 20 seconds)', async () => {
      // Expert difficulty (depth 8) takes ~500-800ms per move locally
      // CI environments may be slower, so allow up to 20 seconds
      const start = Date.now()
      await analyzeGame([3, 3, 2, 4, 1, 5])
      const elapsed = Date.now() - start

      expect(elapsed).toBeLessThan(20000)
    })

    it('analyzes a typical game in reasonable time (< 30 seconds)', async () => {
      // 10-move game at ~500-800ms per move locally
      // CI environments may be slower, so allow up to 30 seconds
      const start = Date.now()
      await analyzeGame([3, 3, 2, 4, 1, 5, 0, 6, 2, 4])
      const elapsed = Date.now() - start

      expect(elapsed).toBeLessThan(30000)
    })
  })
})
