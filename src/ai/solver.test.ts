import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  encodePosition,
  decodePosition,
  scoreToValue,
  scoreToDistance,
  queryPosition,
  analyzeWithSolver,
  getOptimalMove,
  canSolvePosition,
  clearSolverCache,
  getSolverCacheSize,
  boardToDebugString,
  describeSolverResult,
  type SolverAnalysis,
} from './solver'
import { createEmptyBoard } from '../game/makefour'
import type { Position } from './coach'

// Mock fetch for testing
const mockFetch = vi.fn()

beforeEach(() => {
  vi.stubGlobal('fetch', mockFetch)
  clearSolverCache()
})

afterEach(() => {
  vi.unstubAllGlobals()
  mockFetch.mockReset()
})

describe('Solver - Position Encoding', () => {
  describe('encodePosition', () => {
    it('encodes empty move history as empty string', () => {
      expect(encodePosition([])).toBe('')
    })

    it('converts 0-indexed columns to 1-indexed', () => {
      expect(encodePosition([0])).toBe('1')
      expect(encodePosition([6])).toBe('7')
      expect(encodePosition([3])).toBe('4')
    })

    it('encodes multiple moves correctly', () => {
      // Opening: center, center, left of center
      expect(encodePosition([3, 3, 2])).toBe('443')
    })

    it('encodes a full game', () => {
      // Example game sequence
      const moves = [3, 3, 2, 4, 1, 5, 0] // 7-move game
      expect(encodePosition(moves)).toBe('4435261')
    })
  })

  describe('decodePosition', () => {
    it('decodes empty string to empty array', () => {
      expect(decodePosition('')).toEqual([])
    })

    it('converts 1-indexed to 0-indexed', () => {
      expect(decodePosition('1')).toEqual([0])
      expect(decodePosition('7')).toEqual([6])
      expect(decodePosition('4')).toEqual([3])
    })

    it('decodes multiple moves', () => {
      expect(decodePosition('443')).toEqual([3, 3, 2])
    })

    it('round-trips with encodePosition', () => {
      const original = [3, 3, 2, 4, 1, 5, 0]
      const encoded = encodePosition(original)
      const decoded = decodePosition(encoded)
      expect(decoded).toEqual(original)
    })
  })
})

describe('Solver - Score Interpretation', () => {
  describe('scoreToValue', () => {
    it('positive scores indicate win', () => {
      expect(scoreToValue(1)).toBe('win')
      expect(scoreToValue(10)).toBe('win')
      expect(scoreToValue(21)).toBe('win')
    })

    it('negative scores indicate loss', () => {
      expect(scoreToValue(-1)).toBe('loss')
      expect(scoreToValue(-10)).toBe('loss')
      expect(scoreToValue(-21)).toBe('loss')
    })

    it('zero indicates draw', () => {
      expect(scoreToValue(0)).toBe('draw')
    })
  })

  describe('scoreToDistance', () => {
    it('returns undefined for draw (score 0)', () => {
      expect(scoreToDistance(0)).toBeUndefined()
    })

    it('calculates distance for winning scores', () => {
      // Score of 21 means win in 1 move (21 + 1 - 21 = 1)
      expect(scoreToDistance(21)).toBe(1)
      // Score of 20 means win in 2 moves
      expect(scoreToDistance(20)).toBe(2)
      // Score of 10 means win in 12 moves
      expect(scoreToDistance(10)).toBe(12)
    })

    it('calculates distance for losing scores', () => {
      // Score of -21 means loss in 1 move
      expect(scoreToDistance(-21)).toBe(1)
      // Score of -10 means loss in 12 moves
      expect(scoreToDistance(-10)).toBe(12)
    })
  })
})

describe('Solver - API Client', () => {
  const createTestPosition = (moveHistory: number[] = []): Position => ({
    board: createEmptyBoard(),
    currentPlayer: moveHistory.length % 2 === 0 ? 1 : 2,
    moveHistory,
  })

  describe('queryPosition', () => {
    it('returns successful result on valid API response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '4',
          score: [3, 3, 3, 5, 3, 3, 3], // Center column is best
        }),
      })

      const position = createTestPosition([3]) // One move played
      const result = await queryPosition(position, 1000)

      expect(result.success).toBe(true)
      expect(result.score).toBe(5) // Best move score
      expect(result.moveScores[3]).toBe(5) // Center
    })

    it('handles column 100 as invalid/full', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '',
          score: [100, 1, 2, 3, 2, 1, 100], // Columns 0 and 6 invalid
        }),
      })

      const position = createTestPosition()
      const result = await queryPosition(position, 1000)

      expect(result.success).toBe(true)
      expect(result.moveScores[0]).toBeNull()
      expect(result.moveScores[6]).toBeNull()
      expect(result.moveScores[3]).toBe(3)
    })

    it('returns error on API failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      })

      const position = createTestPosition()
      const result = await queryPosition(position, 1000)

      expect(result.success).toBe(false)
      expect(result.error).toContain('500')
    })

    it('handles network timeout', async () => {
      mockFetch.mockImplementation(
        () => new Promise((_, reject) => setTimeout(() => reject(new DOMException('Aborted', 'AbortError')), 100))
      )

      const position = createTestPosition()
      const result = await queryPosition(position, 50)

      expect(result.success).toBe(false)
      expect(result.error).toContain('timed out')
    })

    it('caches successful results', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        json: async () => ({
          pos: '4',
          score: [1, 2, 3, 4, 3, 2, 1],
        }),
      })

      const position = createTestPosition([3])

      // First call
      await queryPosition(position, 1000)
      expect(mockFetch).toHaveBeenCalledTimes(1)

      // Second call should use cache
      const result = await queryPosition(position, 1000)
      expect(mockFetch).toHaveBeenCalledTimes(1)
      expect(result.cached).toBe(true)
    })
  })

  describe('analyzeWithSolver', () => {
    it('returns analysis with optimal moves', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '',
          score: [-2, -1, 0, 5, 0, -1, -2], // Column 3 is best
        }),
      })

      const position = createTestPosition()
      const analysis = await analyzeWithSolver(position, 1000)

      expect(analysis).not.toBeNull()
      expect(analysis!.value).toBe('win')
      expect(analysis!.optimalMoves).toEqual([3])
    })

    it('returns multiple optimal moves when tied', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '',
          score: [0, 0, 5, 5, 5, 0, 0], // Columns 2, 3, 4 are tied
        }),
      })

      const position = createTestPosition()
      const analysis = await analyzeWithSolver(position, 1000)

      expect(analysis!.optimalMoves).toContain(2)
      expect(analysis!.optimalMoves).toContain(3)
      expect(analysis!.optimalMoves).toContain(4)
    })

    it('ranks all valid moves', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '',
          score: [1, 2, 3, 4, 3, 2, 1],
        }),
      })

      const position = createTestPosition()
      const analysis = await analyzeWithSolver(position, 1000)

      expect(analysis!.rankedMoves.length).toBe(7)
      expect(analysis!.rankedMoves[0].column).toBe(3) // Highest score
      expect(analysis!.rankedMoves[0].score).toBe(4)
    })

    it('returns null on API failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      })

      const position = createTestPosition()
      const analysis = await analyzeWithSolver(position, 1000)

      expect(analysis).toBeNull()
    })
  })

  describe('getOptimalMove', () => {
    it('returns optimal column', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '',
          score: [0, 0, 0, 5, 0, 0, 0], // Only column 3 is optimal
        }),
      })

      const position = createTestPosition()
      const move = await getOptimalMove(position, 1000)

      expect(move).toBe(3)
    })

    it('prefers center when multiple optimal moves exist', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '',
          score: [0, 5, 0, 5, 0, 5, 0], // Columns 1, 3, 5 are tied
        }),
      })

      const position = createTestPosition()
      const move = await getOptimalMove(position, 1000)

      expect(move).toBe(3) // Center preferred
    })

    it('returns null on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      })

      const position = createTestPosition()
      const move = await getOptimalMove(position, 1000)

      expect(move).toBeNull()
    })
  })

  describe('canSolvePosition', () => {
    it('returns true on successful query', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          pos: '',
          score: [0, 0, 0, 0, 0, 0, 0],
        }),
      })

      const position = createTestPosition()
      const canSolve = await canSolvePosition(position, 1000)

      expect(canSolve).toBe(true)
    })

    it('returns false on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
      })

      const position = createTestPosition()
      const canSolve = await canSolvePosition(position, 1000)

      expect(canSolve).toBe(false)
    })
  })
})

describe('Solver - Cache Management', () => {
  it('clearSolverCache empties the cache', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        pos: '',
        score: [0, 0, 0, 0, 0, 0, 0],
      }),
    })

    const position: Position = {
      board: createEmptyBoard(),
      currentPlayer: 1,
      moveHistory: [],
    }

    await queryPosition(position, 1000)
    expect(getSolverCacheSize()).toBe(1)

    clearSolverCache()
    expect(getSolverCacheSize()).toBe(0)
  })

  it('getSolverCacheSize returns current size', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        pos: '',
        score: [0, 0, 0, 0, 0, 0, 0],
      }),
    })

    expect(getSolverCacheSize()).toBe(0)

    // Cache position 1
    await queryPosition({ board: createEmptyBoard(), currentPlayer: 1, moveHistory: [] }, 1000)
    expect(getSolverCacheSize()).toBe(1)

    // Cache position 2 (different position)
    await queryPosition({ board: createEmptyBoard(), currentPlayer: 2, moveHistory: [3] }, 1000)
    expect(getSolverCacheSize()).toBe(2)
  })
})

describe('Solver - Utility Functions', () => {
  describe('boardToDebugString', () => {
    it('converts empty board to dots', () => {
      const board = createEmptyBoard()
      const str = boardToDebugString(board)
      expect(str).toBe('.......\n.......\n.......\n.......\n.......\n.......')
    })

    it('shows player pieces', () => {
      const board = createEmptyBoard()
      board[5][3] = 1
      board[5][4] = 2
      const str = boardToDebugString(board)
      expect(str).toContain('...12..')
    })
  })

  describe('describeSolverResult', () => {
    it('describes winning position', () => {
      const analysis: SolverAnalysis = {
        value: 'win',
        score: 10,
        optimalMoves: [3],
        rankedMoves: [{ column: 3, score: 10, value: 'win' }],
        distanceToEnd: 12,
      }
      const desc = describeSolverResult(analysis)
      expect(desc).toContain('Winning')
      expect(desc).toContain('12 moves')
      expect(desc).toContain('column 4') // 1-indexed
    })

    it('describes losing position', () => {
      const analysis: SolverAnalysis = {
        value: 'loss',
        score: -10,
        optimalMoves: [3],
        rankedMoves: [{ column: 3, score: -10, value: 'loss' }],
        distanceToEnd: 5,
      }
      const desc = describeSolverResult(analysis)
      expect(desc).toContain('Losing')
      expect(desc).toContain('5 moves')
    })

    it('describes drawn position', () => {
      const analysis: SolverAnalysis = {
        value: 'draw',
        score: 0,
        optimalMoves: [3, 4],
        rankedMoves: [
          { column: 3, score: 0, value: 'draw' },
          { column: 4, score: 0, value: 'draw' },
        ],
      }
      const desc = describeSolverResult(analysis)
      expect(desc).toContain('Drawn')
    })
  })
})
