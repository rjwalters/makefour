/**
 * MakeFour Game Engine
 *
 * A pure TypeScript implementation of the four-in-a-row game logic.
 * This module contains no UI dependencies and can be used on both
 * client and server.
 */

// Board dimensions
export const ROWS = 6
export const COLUMNS = 7
export const WIN_LENGTH = 4

// Player identifiers
export type Player = 1 | 2
export type Cell = Player | null

// Board is represented as a 2D array: board[row][column]
// Row 0 is the TOP of the board (where pieces fall from)
// Row 5 is the BOTTOM of the board (where pieces land first)
export type Board = Cell[][]

// Game state
export type GameResult = Player | 'draw' | null

export interface GameState {
  board: Board
  currentPlayer: Player
  winner: GameResult
  moveHistory: number[] // Array of column indices
}

/**
 * Creates an empty game board.
 * All cells are initialized to null (empty).
 */
export function createEmptyBoard(): Board {
  return Array.from({ length: ROWS }, () => Array(COLUMNS).fill(null))
}

/**
 * Creates a new game state with an empty board.
 * Player 1 always goes first.
 */
export function createGameState(): GameState {
  return {
    board: createEmptyBoard(),
    currentPlayer: 1,
    winner: null,
    moveHistory: [],
  }
}

/**
 * Deep clones a board to avoid mutations.
 */
export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row])
}

/**
 * Gets the row index where a piece would land if dropped in the given column.
 * Returns -1 if the column is full.
 *
 * @param board - The current board state
 * @param column - The column to check (0-6)
 * @returns The row index (0-5) where the piece lands, or -1 if full
 */
export function getAvailableRow(board: Board, column: number): number {
  if (column < 0 || column >= COLUMNS) {
    return -1
  }

  // Start from the bottom row and find the first empty cell
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row][column] === null) {
      return row
    }
  }

  return -1 // Column is full
}

/**
 * Checks if a move is valid (column exists and has space).
 */
export function isValidMove(board: Board, column: number): boolean {
  return getAvailableRow(board, column) !== -1
}

/**
 * Returns an array of valid column indices for the current board state.
 */
export function getValidMoves(board: Board): number[] {
  const moves: number[] = []
  for (let col = 0; col < COLUMNS; col++) {
    if (isValidMove(board, col)) {
      moves.push(col)
    }
  }
  return moves
}

/**
 * Checks if the board is completely full (draw condition).
 */
export function isBoardFull(board: Board): boolean {
  return getValidMoves(board).length === 0
}

/**
 * Applies a move to the board, returning a new board state.
 * Does NOT mutate the original board.
 *
 * @param board - The current board
 * @param column - The column to drop the piece (0-6)
 * @param player - The player making the move (1 or 2)
 * @returns Object with success status and new board (or null if invalid)
 */
export function applyMove(
  board: Board,
  column: number,
  player: Player
): { success: boolean; board: Board | null; row: number } {
  const row = getAvailableRow(board, column)

  if (row === -1) {
    return { success: false, board: null, row: -1 }
  }

  const newBoard = cloneBoard(board)
  newBoard[row][column] = player

  return { success: true, board: newBoard, row }
}

/**
 * Checks for a winner on the board.
 * Checks horizontal, vertical, and both diagonal directions.
 *
 * @param board - The board to check
 * @returns The winning player (1 or 2), 'draw' if board is full, or null if game continues
 */
export function checkWinner(board: Board): GameResult {
  // Check all directions from each cell
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      const cell = board[row][col]
      if (cell === null) continue

      // Check horizontal (right)
      if (col + WIN_LENGTH - 1 < COLUMNS) {
        if (checkLine(board, row, col, 0, 1, cell)) {
          return cell
        }
      }

      // Check vertical (down)
      if (row + WIN_LENGTH - 1 < ROWS) {
        if (checkLine(board, row, col, 1, 0, cell)) {
          return cell
        }
      }

      // Check diagonal (down-right)
      if (row + WIN_LENGTH - 1 < ROWS && col + WIN_LENGTH - 1 < COLUMNS) {
        if (checkLine(board, row, col, 1, 1, cell)) {
          return cell
        }
      }

      // Check diagonal (down-left)
      if (row + WIN_LENGTH - 1 < ROWS && col - WIN_LENGTH + 1 >= 0) {
        if (checkLine(board, row, col, 1, -1, cell)) {
          return cell
        }
      }
    }
  }

  // No winner found - check for draw
  if (isBoardFull(board)) {
    return 'draw'
  }

  return null // Game continues
}

/**
 * Helper: Checks if WIN_LENGTH consecutive cells match the given player.
 */
function checkLine(
  board: Board,
  startRow: number,
  startCol: number,
  deltaRow: number,
  deltaCol: number,
  player: Player
): boolean {
  for (let i = 0; i < WIN_LENGTH; i++) {
    const row = startRow + i * deltaRow
    const col = startCol + i * deltaCol
    if (board[row][col] !== player) {
      return false
    }
  }
  return true
}

/**
 * Gets the winning cells (for highlighting).
 * Returns an array of [row, col] tuples, or null if no winner.
 */
export function getWinningCells(board: Board): [number, number][] | null {
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      const cell = board[row][col]
      if (cell === null) continue

      // Check horizontal
      if (col + WIN_LENGTH - 1 < COLUMNS) {
        if (checkLine(board, row, col, 0, 1, cell)) {
          return Array.from({ length: WIN_LENGTH }, (_, i) => [row, col + i] as [number, number])
        }
      }

      // Check vertical
      if (row + WIN_LENGTH - 1 < ROWS) {
        if (checkLine(board, row, col, 1, 0, cell)) {
          return Array.from({ length: WIN_LENGTH }, (_, i) => [row + i, col] as [number, number])
        }
      }

      // Check diagonal down-right
      if (row + WIN_LENGTH - 1 < ROWS && col + WIN_LENGTH - 1 < COLUMNS) {
        if (checkLine(board, row, col, 1, 1, cell)) {
          return Array.from({ length: WIN_LENGTH }, (_, i) => [row + i, col + i] as [number, number])
        }
      }

      // Check diagonal down-left
      if (row + WIN_LENGTH - 1 < ROWS && col - WIN_LENGTH + 1 >= 0) {
        if (checkLine(board, row, col, 1, -1, cell)) {
          return Array.from({ length: WIN_LENGTH }, (_, i) => [row + i, col - i] as [number, number])
        }
      }
    }
  }

  return null
}

/**
 * Makes a move and returns the updated game state.
 * This is the main function for game play.
 *
 * @param state - The current game state
 * @param column - The column to drop the piece (0-6)
 * @returns Updated game state, or null if move is invalid
 */
export function makeMove(state: GameState, column: number): GameState | null {
  // Can't move if game is already over
  if (state.winner !== null) {
    return null
  }

  const result = applyMove(state.board, column, state.currentPlayer)

  if (!result.success || result.board === null) {
    return null
  }

  const winner = checkWinner(result.board)
  const nextPlayer: Player = state.currentPlayer === 1 ? 2 : 1

  return {
    board: result.board,
    currentPlayer: winner === null ? nextPlayer : state.currentPlayer,
    winner,
    moveHistory: [...state.moveHistory, column],
  }
}

/**
 * Replays a game from a list of moves.
 * Useful for reconstructing game state from stored move history.
 *
 * @param moves - Array of column indices representing each move
 * @returns The final game state, or null if moves are invalid
 */
export function replayMoves(moves: number[]): GameState | null {
  let state = createGameState()

  for (const column of moves) {
    const newState = makeMove(state, column)
    if (newState === null) {
      return null // Invalid move in sequence
    }
    state = newState
  }

  return state
}

/**
 * Gets the game state at a specific move index.
 * Useful for implementing replay controls.
 *
 * @param moves - Full move history
 * @param moveIndex - Number of moves to replay (0 = empty board)
 * @returns Game state at that point, or null if invalid
 */
export function getStateAtMove(moves: number[], moveIndex: number): GameState | null {
  if (moveIndex < 0 || moveIndex > moves.length) {
    return null
  }
  return replayMoves(moves.slice(0, moveIndex))
}
