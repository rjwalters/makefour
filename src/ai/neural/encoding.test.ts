/**
 * Tests for Position Encoding Module
 */

import { describe, it, expect } from 'vitest'
import {
  encodeOneHot,
  encodeBitboard,
  encodeFlatBinary,
  encodePosition,
  decodeOneHot,
  maskInvalidMoves,
  sampleFromPolicy,
  getInputShape,
  getInputSize,
} from './encoding'
import { createGameState, makeMove } from '../../game/makefour'

describe('Position Encoding', () => {
  describe('encodeOneHot', () => {
    it('should encode empty board correctly', () => {
      const state = createGameState()
      const encoded = encodeOneHot(state)

      expect(encoded.shape).toEqual([1, 3, 6, 7])
      expect(encoded.input.length).toBe(126) // 3 * 6 * 7
      expect(encoded.currentPlayer).toBe(1)
      expect(encoded.moveCount).toBe(0)

      // Channel 0 (player 1) should be all zeros
      for (let i = 0; i < 42; i++) {
        expect(encoded.input[i]).toBe(0)
      }

      // Channel 1 (player 2) should be all zeros
      for (let i = 42; i < 84; i++) {
        expect(encoded.input[i]).toBe(0)
      }

      // Channel 2 (current player = 1) should be all ones
      for (let i = 84; i < 126; i++) {
        expect(encoded.input[i]).toBe(1)
      }
    })

    it('should encode position with pieces correctly', () => {
      let state = createGameState()
      state = makeMove(state, 3)! // Player 1 in column 3
      state = makeMove(state, 4)! // Player 2 in column 4
      state = makeMove(state, 3)! // Player 1 in column 3 again

      const encoded = encodeOneHot(state)

      // Player 2 should be current (3 moves made)
      expect(encoded.currentPlayer).toBe(2)
      expect(encoded.moveCount).toBe(3)

      // Check player 1 pieces (bottom of column 3, and one above)
      // Row 5, Col 3 -> index 5*7+3 = 38
      // Row 4, Col 3 -> index 4*7+3 = 31
      expect(encoded.input[38]).toBe(1) // Player 1 piece
      expect(encoded.input[31]).toBe(1) // Player 1 piece

      // Check player 2 piece (bottom of column 4)
      // Row 5, Col 4 -> 42 + 5*7+4 = 42 + 39 = 81
      expect(encoded.input[81]).toBe(1) // Player 2 piece

      // Channel 2 should be all zeros (player 2's turn)
      for (let i = 84; i < 126; i++) {
        expect(encoded.input[i]).toBe(0)
      }
    })
  })

  describe('encodeFlatBinary', () => {
    it('should encode empty board correctly', () => {
      const state = createGameState()
      const encoded = encodeFlatBinary(state)

      expect(encoded.shape).toEqual([1, 85])
      expect(encoded.input.length).toBe(85)
      expect(encoded.currentPlayer).toBe(1)

      // All piece bits should be 0
      for (let i = 0; i < 84; i++) {
        expect(encoded.input[i]).toBe(0)
      }

      // Current player bit should be 1 (player 1)
      expect(encoded.input[84]).toBe(1)
    })

    it('should encode position with pieces', () => {
      let state = createGameState()
      state = makeMove(state, 0)! // Player 1 in column 0

      const encoded = encodeFlatBinary(state)

      // Player 1 piece at row 5, col 0 -> index 35
      expect(encoded.input[35]).toBe(1)

      // Player 2's turn, so last bit is 0
      expect(encoded.input[84]).toBe(0)
    })
  })

  describe('encodeBitboard', () => {
    it('should encode empty board correctly', () => {
      const state = createGameState()
      const encoded = encodeBitboard(state)

      expect(encoded.shape).toEqual([1, 4])
      expect(encoded.input.length).toBe(4)

      expect(encoded.input[0]).toBe(0) // P1 bitboard
      expect(encoded.input[1]).toBe(0) // P2 bitboard
      expect(encoded.input[2]).toBe(1) // Current player is 1
      expect(encoded.input[3]).toBe(0) // Move count
    })
  })

  describe('encodePosition', () => {
    it('should dispatch to correct encoder', () => {
      const state = createGameState()

      const oneHot = encodePosition(state, 'onehot-6x7x3')
      expect(oneHot.shape).toEqual([1, 3, 6, 7])

      const flatBinary = encodePosition(state, 'flat-binary')
      expect(flatBinary.shape).toEqual([1, 85])

      const bitboard = encodePosition(state, 'bitboard')
      expect(bitboard.shape).toEqual([1, 4])
    })

    it('should throw for unknown encoding', () => {
      const state = createGameState()
      expect(() => encodePosition(state, 'unknown' as any)).toThrow('Unknown encoding type')
    })
  })

  describe('decodeOneHot', () => {
    it('should decode empty board', () => {
      const state = createGameState()
      const encoded = encodeOneHot(state)
      const decoded = decodeOneHot(encoded)

      expect(decoded.currentPlayer).toBe(1)
      // All cells should be null
      for (let row = 0; row < 6; row++) {
        for (let col = 0; col < 7; col++) {
          expect(decoded.board[row][col]).toBe(null)
        }
      }
    })

    it('should decode position with pieces', () => {
      let state = createGameState()
      state = makeMove(state, 3)!
      state = makeMove(state, 4)!

      const encoded = encodeOneHot(state)
      const decoded = decodeOneHot(encoded)

      expect(decoded.currentPlayer).toBe(1) // Back to player 1
      expect(decoded.board[5][3]).toBe(1) // Player 1 piece
      expect(decoded.board[5][4]).toBe(2) // Player 2 piece
    })
  })

  describe('maskInvalidMoves', () => {
    it('should mask full columns', () => {
      // Create a board with column 0 full by both players playing in column 0
      // This results in an alternating column: P1, P2, P1, P2, P1, P2 (no 4-in-a-row)
      let state = createGameState()
      state = makeMove(state, 0)! // P1 in col 0 (row 5)
      state = makeMove(state, 0)! // P2 in col 0 (row 4)
      state = makeMove(state, 0)! // P1 in col 0 (row 3)
      state = makeMove(state, 0)! // P2 in col 0 (row 2)
      state = makeMove(state, 0)! // P1 in col 0 (row 1)
      state = makeMove(state, 0)! // P2 in col 0 (row 0) - column 0 now full!

      const rawPolicy = [0.2, 0.1, 0.1, 0.2, 0.15, 0.15, 0.1]
      const masked = maskInvalidMoves(rawPolicy, state.board)

      // Column 0 should be masked to 0
      expect(masked[0]).toBe(0)

      // Other probabilities should be normalized
      const sum = masked.reduce((a, b) => a + b, 0)
      expect(sum).toBeCloseTo(1, 5)
    })

    it('should handle all valid columns', () => {
      const state = createGameState()
      const rawPolicy = [0.1, 0.1, 0.2, 0.3, 0.15, 0.1, 0.05]
      const masked = maskInvalidMoves(rawPolicy, state.board)

      const sum = masked.reduce((a, b) => a + b, 0)
      expect(sum).toBeCloseTo(1, 5)
    })

    it('should ensure non-negative probabilities', () => {
      const state = createGameState()
      const rawPolicy = [-0.1, 0.2, 0.3, 0.4, 0.1, 0.05, 0.05]
      const masked = maskInvalidMoves(rawPolicy, state.board)

      // Negative value should be clamped to 0
      expect(masked[0]).toBe(0)
    })
  })

  describe('sampleFromPolicy', () => {
    it('should return argmax with temperature 0', () => {
      const policy = [0.1, 0.1, 0.1, 0.5, 0.1, 0.05, 0.05]
      const move = sampleFromPolicy(policy, 0)
      expect(move).toBe(3) // Highest probability
    })

    it('should handle ties in argmax', () => {
      const policy = [0.5, 0.5, 0, 0, 0, 0, 0]
      const move = sampleFromPolicy(policy, 0)
      expect(move).toBe(0) // First max
    })

    it('should return valid column with temperature > 0', () => {
      const policy = [0.1, 0.2, 0.2, 0.3, 0.1, 0.05, 0.05]
      const move = sampleFromPolicy(policy, 1.0)
      expect(move).toBeGreaterThanOrEqual(0)
      expect(move).toBeLessThanOrEqual(6)
    })
  })

  describe('getInputShape', () => {
    it('should return correct shapes', () => {
      expect(getInputShape('onehot-6x7x3')).toEqual([1, 3, 6, 7])
      expect(getInputShape('flat-binary')).toEqual([1, 85])
      expect(getInputShape('bitboard')).toEqual([1, 4])
    })
  })

  describe('getInputSize', () => {
    it('should return correct sizes', () => {
      expect(getInputSize('onehot-6x7x3')).toBe(126)
      expect(getInputSize('flat-binary')).toBe(85)
      expect(getInputSize('bitboard')).toBe(4)
    })
  })
})
