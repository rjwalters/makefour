/**
 * Position Encoding for Neural Network Input
 *
 * Provides different encoding schemes for representing Connect Four positions
 * as neural network inputs.
 */

import { type Board, ROWS, COLUMNS, type Player, type GameState } from '../../game/makefour'
import type { EncodedPosition, PositionEncodingType } from './interface'

/**
 * Encodes a position using one-hot encoding with 3 channels.
 * Shape: [1, 3, 6, 7] = [batch, channels, rows, columns]
 *
 * Channel 0: Player 1's pieces (1 where P1 has a piece, 0 elsewhere)
 * Channel 1: Player 2's pieces (1 where P2 has a piece, 0 elsewhere)
 * Channel 2: Current player indicator (all 1s if P1 to move, all 0s if P2)
 *
 * This encoding preserves spatial relationships for CNNs.
 */
export function encodeOneHot(state: GameState): EncodedPosition {
  const { board, currentPlayer, moveHistory } = state
  const channelSize = ROWS * COLUMNS // 42
  const totalSize = 3 * channelSize // 126

  const input = new Float32Array(totalSize)

  // Channel 0: Player 1 pieces
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (board[row][col] === 1) {
        input[row * COLUMNS + col] = 1
      }
    }
  }

  // Channel 1: Player 2 pieces
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (board[row][col] === 2) {
        input[channelSize + row * COLUMNS + col] = 1
      }
    }
  }

  // Channel 2: Current player (all 1s if player 1 to move)
  if (currentPlayer === 1) {
    for (let i = 0; i < channelSize; i++) {
      input[2 * channelSize + i] = 1
    }
  }

  return {
    input,
    shape: [1, 3, ROWS, COLUMNS],
    currentPlayer,
    moveCount: moveHistory.length,
  }
}

/**
 * Encodes a position using bitboard representation.
 * Shape: [1, 2] where each element is a 49-bit integer represented as two Float32s.
 *
 * Each player's position is represented as a bitmask of occupied cells.
 * Cell (row, col) maps to bit (row * 7 + col).
 */
export function encodeBitboard(state: GameState): EncodedPosition {
  const { board, currentPlayer, moveHistory } = state

  // Use two numbers to represent each 42-bit bitboard
  // (JavaScript numbers are 64-bit floats, safe integers up to 2^53)
  let p1Bits = 0
  let p2Bits = 0

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      const bit = row * COLUMNS + col
      if (board[row][col] === 1) {
        p1Bits |= 1 << bit
      } else if (board[row][col] === 2) {
        p2Bits |= 1 << bit
      }
    }
  }

  // Include current player info
  const playerBit = currentPlayer === 1 ? 1 : 0

  const input = new Float32Array([p1Bits, p2Bits, playerBit, moveHistory.length])

  return {
    input,
    shape: [1, 4],
    currentPlayer,
    moveCount: moveHistory.length,
  }
}

/**
 * Encodes a position as a flat binary vector.
 * Shape: [1, 85] = [batch, features]
 *
 * Features:
 * - 42 bits for player 1 pieces
 * - 42 bits for player 2 pieces
 * - 1 bit for current player
 *
 * Simple and efficient for MLP architectures.
 */
export function encodeFlatBinary(state: GameState): EncodedPosition {
  const { board, currentPlayer, moveHistory } = state
  const cellCount = ROWS * COLUMNS // 42
  const totalSize = cellCount * 2 + 1 // 85

  const input = new Float32Array(totalSize)

  // First 42: Player 1 pieces
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (board[row][col] === 1) {
        input[row * COLUMNS + col] = 1
      }
    }
  }

  // Next 42: Player 2 pieces
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (board[row][col] === 2) {
        input[cellCount + row * COLUMNS + col] = 1
      }
    }
  }

  // Last bit: Current player (1 if player 1, 0 if player 2)
  input[cellCount * 2] = currentPlayer === 1 ? 1 : 0

  return {
    input,
    shape: [1, totalSize],
    currentPlayer,
    moveCount: moveHistory.length,
  }
}

/**
 * Encodes a position using the specified encoding type.
 */
export function encodePosition(
  state: GameState,
  encoding: PositionEncodingType
): EncodedPosition {
  switch (encoding) {
    case 'onehot-6x7x3':
      return encodeOneHot(state)
    case 'bitboard':
      return encodeBitboard(state)
    case 'flat-binary':
      return encodeFlatBinary(state)
    default:
      throw new Error(`Unknown encoding type: ${encoding}`)
  }
}

/**
 * Decodes a one-hot encoded position back to a board state.
 * Useful for debugging and visualization.
 */
export function decodeOneHot(encoded: EncodedPosition): { board: Board; currentPlayer: Player } {
  if (encoded.shape[1] !== 3 || encoded.shape[2] !== ROWS || encoded.shape[3] !== COLUMNS) {
    throw new Error('Invalid one-hot encoding shape')
  }

  const channelSize = ROWS * COLUMNS
  const board: Board = Array.from({ length: ROWS }, () => Array(COLUMNS).fill(null))

  // Decode player 1 pieces
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (encoded.input[row * COLUMNS + col] === 1) {
        board[row][col] = 1
      }
    }
  }

  // Decode player 2 pieces
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      if (encoded.input[channelSize + row * COLUMNS + col] === 1) {
        board[row][col] = 2
      }
    }
  }

  // Decode current player
  const currentPlayer: Player = encoded.input[2 * channelSize] === 1 ? 1 : 2

  return { board, currentPlayer }
}

/**
 * Converts neural network output probabilities to valid move probabilities.
 * Masks out invalid moves (full columns) and renormalizes.
 *
 * @param rawPolicy - Raw policy output from neural network (7 values)
 * @param board - Current board state
 * @returns Normalized probabilities for valid moves only
 */
export function maskInvalidMoves(rawPolicy: number[], board: Board): number[] {
  const masked = rawPolicy.map((prob, col) => {
    // Check if column is full (top row occupied)
    if (board[0][col] !== null) {
      return 0
    }
    return Math.max(0, prob) // Ensure non-negative
  })

  // Renormalize
  const sum = masked.reduce((a, b) => a + b, 0)
  if (sum === 0) {
    // No valid moves - shouldn't happen in normal gameplay
    return masked
  }

  return masked.map((p) => p / sum)
}

/**
 * Selects a move from policy probabilities.
 *
 * @param policy - Move probabilities (7 values)
 * @param temperature - Temperature for sampling (0 = deterministic, higher = more random)
 * @returns Selected column index (0-6)
 */
export function sampleFromPolicy(policy: number[], temperature = 0): number {
  if (temperature === 0) {
    // Deterministic: pick highest probability
    let maxProb = -1
    let maxCol = 0
    for (let col = 0; col < policy.length; col++) {
      if (policy[col] > maxProb) {
        maxProb = policy[col]
        maxCol = col
      }
    }
    return maxCol
  }

  // Temperature-scaled sampling
  const scaled = policy.map((p) => p ** (1 / temperature))
  const sum = scaled.reduce((a, b) => a + b, 0)
  const normalized = scaled.map((p) => p / sum)

  // Random sampling
  const rand = Math.random()
  let cumulative = 0
  for (let col = 0; col < normalized.length; col++) {
    cumulative += normalized[col]
    if (rand < cumulative) {
      return col
    }
  }

  return normalized.length - 1
}

/**
 * Gets the input shape for a given encoding type.
 */
export function getInputShape(encoding: PositionEncodingType): number[] {
  switch (encoding) {
    case 'onehot-6x7x3':
      return [1, 3, ROWS, COLUMNS]
    case 'bitboard':
      return [1, 4]
    case 'flat-binary':
      return [1, 85]
    default:
      throw new Error(`Unknown encoding type: ${encoding}`)
  }
}

/**
 * Gets the total number of input features for a given encoding type.
 */
export function getInputSize(encoding: PositionEncodingType): number {
  const shape = getInputShape(encoding)
  return shape.reduce((a, b) => a * b, 1)
}
