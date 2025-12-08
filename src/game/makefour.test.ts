import { describe, it, expect } from 'vitest'
import {
  ROWS,
  COLUMNS,
  createEmptyBoard,
  createGameState,
  getAvailableRow,
  isValidMove,
  getValidMoves,
  isBoardFull,
  applyMove,
  checkWinner,
  getWinningCells,
  makeMove,
  replayMoves,
  getStateAtMove,
  type Board,
  type Player,
} from './makefour'

describe('MakeFour Game Engine', () => {
  describe('createEmptyBoard', () => {
    it('creates a 6x7 board', () => {
      const board = createEmptyBoard()
      expect(board.length).toBe(ROWS)
      expect(board[0].length).toBe(COLUMNS)
    })

    it('all cells are null', () => {
      const board = createEmptyBoard()
      for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLUMNS; col++) {
          expect(board[row][col]).toBeNull()
        }
      }
    })
  })

  describe('createGameState', () => {
    it('creates initial game state', () => {
      const state = createGameState()
      expect(state.currentPlayer).toBe(1)
      expect(state.winner).toBeNull()
      expect(state.moveHistory).toEqual([])
    })
  })

  describe('getAvailableRow', () => {
    it('returns bottom row for empty column', () => {
      const board = createEmptyBoard()
      expect(getAvailableRow(board, 3)).toBe(ROWS - 1) // Row 5
    })

    it('returns next available row after pieces', () => {
      const board = createEmptyBoard()
      board[5][3] = 1 // Bottom piece
      expect(getAvailableRow(board, 3)).toBe(4)

      board[4][3] = 2
      expect(getAvailableRow(board, 3)).toBe(3)
    })

    it('returns -1 for full column', () => {
      const board = createEmptyBoard()
      for (let row = 0; row < ROWS; row++) {
        board[row][3] = ((row % 2) + 1) as Player
      }
      expect(getAvailableRow(board, 3)).toBe(-1)
    })

    it('returns -1 for invalid column', () => {
      const board = createEmptyBoard()
      expect(getAvailableRow(board, -1)).toBe(-1)
      expect(getAvailableRow(board, 7)).toBe(-1)
    })
  })

  describe('isValidMove', () => {
    it('returns true for empty column', () => {
      const board = createEmptyBoard()
      expect(isValidMove(board, 3)).toBe(true)
    })

    it('returns false for full column', () => {
      const board = createEmptyBoard()
      for (let row = 0; row < ROWS; row++) {
        board[row][3] = 1
      }
      expect(isValidMove(board, 3)).toBe(false)
    })
  })

  describe('getValidMoves', () => {
    it('returns all columns for empty board', () => {
      const board = createEmptyBoard()
      expect(getValidMoves(board)).toEqual([0, 1, 2, 3, 4, 5, 6])
    })

    it('excludes full columns', () => {
      const board = createEmptyBoard()
      // Fill column 3
      for (let row = 0; row < ROWS; row++) {
        board[row][3] = 1
      }
      expect(getValidMoves(board)).toEqual([0, 1, 2, 4, 5, 6])
    })
  })

  describe('isBoardFull', () => {
    it('returns false for empty board', () => {
      const board = createEmptyBoard()
      expect(isBoardFull(board)).toBe(false)
    })

    it('returns true when all columns are full', () => {
      const board = createEmptyBoard()
      for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLUMNS; col++) {
          board[row][col] = ((row + col) % 2 + 1) as Player
        }
      }
      expect(isBoardFull(board)).toBe(true)
    })
  })

  describe('applyMove', () => {
    it('places piece at bottom of empty column', () => {
      const board = createEmptyBoard()
      const result = applyMove(board, 3, 1)
      expect(result.success).toBe(true)
      expect(result.row).toBe(5)
      expect(result.board![5][3]).toBe(1)
    })

    it('does not mutate original board', () => {
      const board = createEmptyBoard()
      applyMove(board, 3, 1)
      expect(board[5][3]).toBeNull()
    })

    it('stacks pieces correctly', () => {
      let board = createEmptyBoard()
      let result = applyMove(board, 3, 1)
      board = result.board!

      result = applyMove(board, 3, 2)
      expect(result.row).toBe(4)
      expect(result.board![4][3]).toBe(2)
    })

    it('fails for full column', () => {
      const board = createEmptyBoard()
      for (let row = 0; row < ROWS; row++) {
        board[row][3] = 1
      }
      const result = applyMove(board, 3, 2)
      expect(result.success).toBe(false)
      expect(result.board).toBeNull()
    })
  })

  describe('checkWinner', () => {
    it('returns null for empty board', () => {
      const board = createEmptyBoard()
      expect(checkWinner(board)).toBeNull()
    })

    it('detects horizontal win', () => {
      const board = createEmptyBoard()
      // Player 1 wins horizontally on bottom row
      board[5][0] = 1
      board[5][1] = 1
      board[5][2] = 1
      board[5][3] = 1
      expect(checkWinner(board)).toBe(1)
    })

    it('detects vertical win', () => {
      const board = createEmptyBoard()
      // Player 2 wins vertically
      board[5][0] = 2
      board[4][0] = 2
      board[3][0] = 2
      board[2][0] = 2
      expect(checkWinner(board)).toBe(2)
    })

    it('detects diagonal win (down-right)', () => {
      const board = createEmptyBoard()
      // Player 1 wins diagonally from top-left to bottom-right
      board[2][0] = 1
      board[3][1] = 1
      board[4][2] = 1
      board[5][3] = 1
      expect(checkWinner(board)).toBe(1)
    })

    it('detects diagonal win (down-left)', () => {
      const board = createEmptyBoard()
      // Player 2 wins diagonally from top-right to bottom-left
      board[2][6] = 2
      board[3][5] = 2
      board[4][4] = 2
      board[5][3] = 2
      expect(checkWinner(board)).toBe(2)
    })

    it('returns draw when board is full with no winner', () => {
      // Create a full board with no four-in-a-row
      // This is a valid draw pattern
      const board: Board = [
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
        [1, 1, 2, 1, 1, 2, 1],
        [2, 2, 1, 2, 2, 1, 2],
      ]
      expect(checkWinner(board)).toBe('draw')
    })

    it('does not detect three in a row as win', () => {
      const board = createEmptyBoard()
      board[5][0] = 1
      board[5][1] = 1
      board[5][2] = 1
      expect(checkWinner(board)).toBeNull()
    })
  })

  describe('getWinningCells', () => {
    it('returns null when no winner', () => {
      const board = createEmptyBoard()
      expect(getWinningCells(board)).toBeNull()
    })

    it('returns correct cells for horizontal win', () => {
      const board = createEmptyBoard()
      board[5][1] = 1
      board[5][2] = 1
      board[5][3] = 1
      board[5][4] = 1
      const cells = getWinningCells(board)
      expect(cells).toEqual([
        [5, 1],
        [5, 2],
        [5, 3],
        [5, 4],
      ])
    })

    it('returns correct cells for vertical win', () => {
      const board = createEmptyBoard()
      board[2][3] = 2
      board[3][3] = 2
      board[4][3] = 2
      board[5][3] = 2
      const cells = getWinningCells(board)
      expect(cells).toEqual([
        [2, 3],
        [3, 3],
        [4, 3],
        [5, 3],
      ])
    })
  })

  describe('makeMove', () => {
    it('updates game state correctly', () => {
      const state = createGameState()
      const newState = makeMove(state, 3)

      expect(newState).not.toBeNull()
      expect(newState!.board[5][3]).toBe(1)
      expect(newState!.currentPlayer).toBe(2)
      expect(newState!.moveHistory).toEqual([3])
    })

    it('alternates players', () => {
      let state = createGameState()
      state = makeMove(state, 0)!
      expect(state.currentPlayer).toBe(2)

      state = makeMove(state, 1)!
      expect(state.currentPlayer).toBe(1)
    })

    it('returns null for invalid move', () => {
      const board = createEmptyBoard()
      for (let row = 0; row < ROWS; row++) {
        board[row][3] = 1
      }
      const state = { ...createGameState(), board }
      expect(makeMove(state, 3)).toBeNull()
    })

    it('returns null if game is already won', () => {
      const state = createGameState()
      const wonState = {
        ...state,
        winner: 1 as const,
      }
      expect(makeMove(wonState, 0)).toBeNull()
    })

    it('detects winner correctly', () => {
      let state = createGameState()
      // Player 1: columns 0, 1, 2, 3 (bottom row)
      // Player 2: columns 0, 1, 2 (second row)
      state = makeMove(state, 0)! // P1
      state = makeMove(state, 0)! // P2
      state = makeMove(state, 1)! // P1
      state = makeMove(state, 1)! // P2
      state = makeMove(state, 2)! // P1
      state = makeMove(state, 2)! // P2
      state = makeMove(state, 3)! // P1 wins

      expect(state.winner).toBe(1)
    })
  })

  describe('replayMoves', () => {
    it('reconstructs game state from moves', () => {
      const moves = [3, 2, 3, 2, 3, 2, 3] // P1 vertical win
      const state = replayMoves(moves)

      expect(state).not.toBeNull()
      expect(state!.winner).toBe(1)
      expect(state!.moveHistory).toEqual(moves)
    })

    it('returns null for invalid move sequence', () => {
      // Try to play 7 moves in same column (only 6 fit)
      const moves = [0, 0, 0, 0, 0, 0, 0]
      expect(replayMoves(moves)).toBeNull()
    })

    it('handles empty moves array', () => {
      const state = replayMoves([])
      expect(state).not.toBeNull()
      expect(state!.moveHistory).toEqual([])
      expect(state!.currentPlayer).toBe(1)
    })
  })

  describe('getStateAtMove', () => {
    it('returns empty board for move 0', () => {
      const moves = [0, 1, 2, 3]
      const state = getStateAtMove(moves, 0)
      expect(state!.moveHistory).toEqual([])
    })

    it('returns correct state for middle of game', () => {
      const moves = [0, 1, 2, 3]
      const state = getStateAtMove(moves, 2)
      expect(state!.moveHistory).toEqual([0, 1])
    })

    it('returns final state for full move count', () => {
      const moves = [0, 1, 2, 3]
      const state = getStateAtMove(moves, 4)
      expect(state!.moveHistory).toEqual(moves)
    })

    it('returns null for negative index', () => {
      expect(getStateAtMove([0, 1], -1)).toBeNull()
    })

    it('returns null for index beyond moves length', () => {
      expect(getStateAtMove([0, 1], 5)).toBeNull()
    })
  })
})
