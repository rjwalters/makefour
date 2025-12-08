import { cn } from '@/lib/utils'
import {
  ROWS,
  COLUMNS,
  type Board,
  type Player,
  type GameResult,
  getWinningCells,
} from '../game/makefour'

interface GameBoardProps {
  board: Board
  currentPlayer: Player
  winner: GameResult
  onColumnClick?: (column: number) => void
  disabled?: boolean
}

/**
 * Renders the MakeFour game board.
 * Supports interactive play and read-only replay mode.
 */
export default function GameBoard({
  board,
  currentPlayer,
  winner,
  onColumnClick,
  disabled = false,
}: GameBoardProps) {
  const winningCells = winner && winner !== 'draw' ? getWinningCells(board) : null
  const winningSet = new Set(winningCells?.map(([r, c]) => `${r},${c}`) ?? [])

  const isInteractive = !disabled && winner === null && onColumnClick

  const handleColumnClick = (column: number) => {
    if (isInteractive) {
      onColumnClick(column)
    }
  }

  return (
    <div className="flex flex-col items-center gap-4">
      {/* Column click targets - invisible buttons above each column */}
      <div className="flex gap-1">
        {Array.from({ length: COLUMNS }, (_, col) => (
          <button
            key={`col-${col}`}
            onClick={() => handleColumnClick(col)}
            disabled={!isInteractive}
            className={cn(
              'w-12 h-8 sm:w-14 sm:h-10 rounded-t-lg transition-colors',
              isInteractive
                ? 'hover:bg-gray-200 dark:hover:bg-gray-700 cursor-pointer'
                : 'cursor-default'
            )}
            aria-label={`Drop piece in column ${col + 1}`}
          >
            {/* Drop indicator */}
            {isInteractive && (
              <div
                className={cn(
                  'w-8 h-8 sm:w-10 sm:h-10 mx-auto rounded-full opacity-0 hover:opacity-50 transition-opacity',
                  currentPlayer === 1 ? 'bg-red-500' : 'bg-yellow-500'
                )}
              />
            )}
          </button>
        ))}
      </div>

      {/* Game board */}
      <div className="bg-blue-600 dark:bg-blue-800 p-2 rounded-lg shadow-lg">
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${COLUMNS}, 1fr)` }}>
          {board.map((row, rowIndex) =>
            row.map((cell, colIndex) => {
              const isWinningCell = winningSet.has(`${rowIndex},${colIndex}`)
              return (
                <button
                  key={`${rowIndex}-${colIndex}`}
                  onClick={() => handleColumnClick(colIndex)}
                  disabled={!isInteractive}
                  className={cn(
                    'w-12 h-12 sm:w-14 sm:h-14 rounded-full transition-all duration-200',
                    'flex items-center justify-center',
                    isInteractive ? 'cursor-pointer' : 'cursor-default',
                    // Cell background (the "hole" in the board)
                    'bg-gray-100 dark:bg-gray-900'
                  )}
                  aria-label={
                    cell === null
                      ? `Empty cell, column ${colIndex + 1}`
                      : `Player ${cell} piece, column ${colIndex + 1}`
                  }
                >
                  {/* Piece */}
                  {cell !== null && (
                    <div
                      className={cn(
                        'w-10 h-10 sm:w-12 sm:h-12 rounded-full transition-all',
                        cell === 1
                          ? 'bg-red-500 shadow-red-600/50'
                          : 'bg-yellow-400 shadow-yellow-500/50',
                        'shadow-lg',
                        // Winning piece animation
                        isWinningCell && 'ring-4 ring-white animate-pulse'
                      )}
                    />
                  )}
                </button>
              )
            })
          )}
        </div>
      </div>

      {/* Player color legend */}
      <div className="flex gap-6 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-red-500" />
          <span>Player 1</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-full bg-yellow-400" />
          <span>Player 2</span>
        </div>
      </div>
    </div>
  )
}
