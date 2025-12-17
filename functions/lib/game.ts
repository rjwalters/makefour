/**
 * Server-side game logic for MakeFour
 * This is a minimal subset of the client-side game engine for move validation.
 */

export const ROWS = 6
export const COLUMNS = 7
export const WIN_LENGTH = 4

export type Player = 1 | 2
export type Cell = Player | null
export type Board = Cell[][]
export type GameResult = Player | 'draw' | null

export interface GameState {
  board: Board
  currentPlayer: Player
  winner: GameResult
  moveHistory: number[]
}

/**
 * Creates an empty game board.
 */
export function createEmptyBoard(): Board {
  return Array.from({ length: ROWS }, () => Array(COLUMNS).fill(null))
}

/**
 * Creates a new game state with an empty board.
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
 * Deep clones a board.
 */
export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row])
}

/**
 * Gets the row where a piece would land in the given column.
 * Returns -1 if the column is full.
 */
export function getAvailableRow(board: Board, column: number): number {
  if (column < 0 || column >= COLUMNS) {
    return -1
  }
  for (let row = ROWS - 1; row >= 0; row--) {
    if (board[row][column] === null) {
      return row
    }
  }
  return -1
}

/**
 * Checks if a move is valid.
 */
export function isValidMove(board: Board, column: number): boolean {
  return getAvailableRow(board, column) !== -1
}

/**
 * Returns all valid column indices where a piece can be dropped.
 */
export function getValidMoves(board: Board): number[] {
  const valid: number[] = []
  for (let col = 0; col < COLUMNS; col++) {
    if (isValidMove(board, col)) {
      valid.push(col)
    }
  }
  return valid
}

/**
 * Checks if the board is completely full.
 */
export function isBoardFull(board: Board): boolean {
  for (let col = 0; col < COLUMNS; col++) {
    if (board[0][col] === null) return false
  }
  return true
}

/**
 * Applies a move to the board, returning a new board state.
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
 * Checks for a winner on the board.
 */
export function checkWinner(board: Board): GameResult {
  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLUMNS; col++) {
      const cell = board[row][col]
      if (cell === null) continue

      // Horizontal
      if (col + WIN_LENGTH - 1 < COLUMNS && checkLine(board, row, col, 0, 1, cell)) {
        return cell
      }
      // Vertical
      if (row + WIN_LENGTH - 1 < ROWS && checkLine(board, row, col, 1, 0, cell)) {
        return cell
      }
      // Diagonal down-right
      if (row + WIN_LENGTH - 1 < ROWS && col + WIN_LENGTH - 1 < COLUMNS && checkLine(board, row, col, 1, 1, cell)) {
        return cell
      }
      // Diagonal down-left
      if (row + WIN_LENGTH - 1 < ROWS && col - WIN_LENGTH + 1 >= 0 && checkLine(board, row, col, 1, -1, cell)) {
        return cell
      }
    }
  }

  if (isBoardFull(board)) {
    return 'draw'
  }

  return null
}

/**
 * Makes a move and returns the updated game state.
 */
export function makeMove(state: GameState, column: number): GameState | null {
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
 */
export function replayMoves(moves: number[]): GameState | null {
  let state = createGameState()
  for (const column of moves) {
    const newState = makeMove(state, column)
    if (newState === null) {
      return null
    }
    state = newState
  }
  return state
}
