import { describe, it, expect } from 'vitest'
import {
  evaluatePosition,
  minimax,
  findBestMove,
  analyzePosition,
  suggestMove,
  rankMoves,
  DIFFICULTY_LEVELS,
  type Position,
} from './coach'
import { createEmptyBoard, type Board } from '../game/makefour'

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

describe('AI Coach - Minimax', () => {
  describe('evaluatePosition', () => {
    it('returns 0 for empty board', () => {
      const board = createEmptyBoard()
      expect(evaluatePosition(board, 1)).toBe(0)
    })

    it('gives positive score for center control', () => {
      const board = createEmptyBoard()
      board[5][3] = 1 // Player 1 in center bottom
      const score = evaluatePosition(board, 1)
      expect(score).toBeGreaterThan(0)
    })

    it('gives negative score when opponent controls center', () => {
      const board = createEmptyBoard()
      board[5][3] = 2 // Player 2 in center bottom
      const score = evaluatePosition(board, 1)
      expect(score).toBeLessThan(0)
    })

    it('scores three-in-a-row highly', () => {
      const board = createEmptyBoard()
      // Player 1 has three in bottom row
      board[5][0] = 1
      board[5][1] = 1
      board[5][2] = 1
      const score = evaluatePosition(board, 1)
      expect(score).toBeGreaterThan(50) // Should be significant
    })

    it('scores opponent threats negatively', () => {
      const board = createEmptyBoard()
      // Player 2 has three in bottom row
      board[5][0] = 2
      board[5][1] = 2
      board[5][2] = 2
      const score = evaluatePosition(board, 1)
      expect(score).toBeLessThan(-50)
    })
  })

  describe('minimax', () => {
    it('returns winning move when available', () => {
      // Player 1 can win by playing column 3
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        111....
      `)

      const result = minimax(board, 4, -Infinity, Infinity, true, 1, 1)
      expect(result.move).toBe(3) // Complete the four
      expect(result.score).toBeGreaterThan(100000) // Winning score
    })

    it('blocks opponent winning move', () => {
      // Player 2 has three in a row, Player 1 must block
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        222....
      `)

      const result = minimax(board, 4, -Infinity, Infinity, true, 1, 1)
      expect(result.move).toBe(3) // Block the win
    })

    it('returns draw score for drawn position', () => {
      // Create a full board that's a draw
      const board: Board = [
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
      ]

      const result = minimax(board, 4, -Infinity, Infinity, true, 1, 1)
      expect(result.score).toBe(0)
      expect(result.move).toBeNull()
    })

    it('prefers faster wins (higher depth remaining)', () => {
      // Both columns can lead to a win, but one is faster
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        2......
        111.2..
      `)

      // Column 3 is immediate win
      const result = findBestMove(board, 1, 6)
      expect(result.move).toBe(3)
    })
  })

  describe('findBestMove', () => {
    it('returns valid move for empty board', () => {
      const board = createEmptyBoard()
      const { move } = findBestMove(board, 1, 4)
      expect(move).toBeGreaterThanOrEqual(0)
      expect(move).toBeLessThanOrEqual(6)
    })

    it('prefers center column on empty board', () => {
      const board = createEmptyBoard()
      const { move } = findBestMove(board, 1, 4)
      expect(move).toBe(3) // Center is best opening
    })

    it('finds winning move in complex position', () => {
      // Player 1 can win with column 2 (vertical)
      // Column 2 has pieces at rows 3,4,5 and row 2 is empty
      const board = createBoardFromString(`
        .......
        .......
        ...1...
        ..12...
        .212...
        1212...
      `)

      const { move } = findBestMove(board, 1, 6)
      expect(move).toBe(2) // Completes vertical four
    })
  })

  describe('analyzePosition', () => {
    it('returns correct analysis for winning position', async () => {
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        111....
      `)
      const position: Position = {
        board,
        currentPlayer: 1,
        moveHistory: [0, 6, 1, 6, 2],
      }

      const analysis = await analyzePosition(position, 'intermediate')
      expect(analysis.bestMove).toBe(3) // Win
      expect(analysis.theoreticalResult).toBe('win')
      expect(analysis.confidence).toBeGreaterThan(0.5)
    })

    it('returns completed game analysis for draw', async () => {
      // Create a full draw board
      const fullBoard: Board = [
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
      ]
      const drawPosition: Position = {
        board: fullBoard,
        currentPlayer: 1,
        moveHistory: [],
      }

      const analysis = await analyzePosition(drawPosition, 'intermediate')
      expect(analysis.bestMove).toBe(-1)
      expect(analysis.theoreticalResult).toBe('draw')
    })
  })

  describe('suggestMove', () => {
    it('returns valid move', async () => {
      const position: Position = {
        board: createEmptyBoard(),
        currentPlayer: 1,
        moveHistory: [],
      }

      // Use beginner for fast test on empty board
      const move = await suggestMove(position, 'beginner')
      expect(move).toBeGreaterThanOrEqual(0)
      expect(move).toBeLessThanOrEqual(6)
    })

    it('finds winning move when available (via findBestMove)', () => {
      // Test move finding without error rate by using findBestMove directly
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        111....
      `)

      // Even at low depth (2), the AI should find the obvious winning move
      const { move } = findBestMove(board, 1, 2)
      expect(move).toBe(3)
    })
  })

  describe('rankMoves', () => {
    it('ranks moves from best to worst', async () => {
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        111....
      `)
      const position: Position = {
        board,
        currentPlayer: 1,
        moveHistory: [],
      }

      const ranked = await rankMoves(position, 'intermediate')
      expect(ranked.length).toBe(7) // All columns available
      expect(ranked[0].column).toBe(3) // Winning move is first
      expect(ranked[0].score).toBeGreaterThan(ranked[1].score)
    })

    it('identifies winning move in comments', async () => {
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        111....
      `)
      const position: Position = {
        board,
        currentPlayer: 1,
        moveHistory: [],
      }

      const ranked = await rankMoves(position, 'intermediate')
      expect(ranked[0].comment).toBe('Winning move!')
    })
  })

  describe('DIFFICULTY_LEVELS', () => {
    it('has increasing search depth', () => {
      expect(DIFFICULTY_LEVELS.beginner.searchDepth).toBeLessThan(
        DIFFICULTY_LEVELS.intermediate.searchDepth
      )
      expect(DIFFICULTY_LEVELS.intermediate.searchDepth).toBeLessThan(
        DIFFICULTY_LEVELS.expert.searchDepth
      )
      expect(DIFFICULTY_LEVELS.expert.searchDepth).toBeLessThan(
        DIFFICULTY_LEVELS.perfect.searchDepth
      )
    })

    it('has decreasing error rates', () => {
      expect(DIFFICULTY_LEVELS.beginner.errorRate).toBeGreaterThan(
        DIFFICULTY_LEVELS.intermediate.errorRate
      )
      expect(DIFFICULTY_LEVELS.intermediate.errorRate).toBeGreaterThan(
        DIFFICULTY_LEVELS.expert.errorRate
      )
      expect(DIFFICULTY_LEVELS.expert.errorRate).toBeGreaterThan(
        DIFFICULTY_LEVELS.perfect.errorRate
      )
    })
  })

  describe('AI plays correctly at different difficulty levels', () => {
    it('beginner AI makes legal moves', async () => {
      const position: Position = {
        board: createEmptyBoard(),
        currentPlayer: 1,
        moveHistory: [],
      }

      const move = await suggestMove(position, 'beginner')
      expect(move).toBeGreaterThanOrEqual(0)
      expect(move).toBeLessThanOrEqual(6)
    })

    it('AI blocks obvious wins (tested via findBestMove)', () => {
      // Player 2 has three in a row at columns 0,1,2 - must block at column 3
      const board = createBoardFromString(`
        .......
        .......
        .......
        .......
        .......
        222....
      `)

      // Use findBestMove directly to test without error rate
      const { move } = findBestMove(board, 1, 4)
      expect(move).toBe(3) // Block the win
    })
  })

  describe('Performance', () => {
    it('beginner responds quickly (< 1 second)', async () => {
      const position: Position = {
        board: createEmptyBoard(),
        currentPlayer: 1,
        moveHistory: [],
      }

      const start = Date.now()
      await suggestMove(position, 'beginner')
      const elapsed = Date.now() - start

      expect(elapsed).toBeLessThan(1000)
    })

    it('intermediate responds quickly (< 1 second)', async () => {
      const position: Position = {
        board: createEmptyBoard(),
        currentPlayer: 1,
        moveHistory: [],
      }

      const start = Date.now()
      await suggestMove(position, 'intermediate')
      const elapsed = Date.now() - start

      expect(elapsed).toBeLessThan(1000)
    })
  })
})
