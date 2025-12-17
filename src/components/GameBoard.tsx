import { cn } from '@/lib/utils'
import {
  COLUMNS,
  type Board,
  type Player,
  type GameResult,
  getWinningCells,
} from '../game/makefour'
import type { Threat } from '../ai/coach'

export type RenderMode = 'visual' | 'text'

interface GameBoardProps {
  board: Board
  currentPlayer: Player
  winner: GameResult
  onColumnClick?: (column: number) => void
  disabled?: boolean
  /** Optional threats to highlight on the board */
  threats?: Threat[]
  /** Whether to show threat highlighting */
  showThreats?: boolean
  /** Render mode: 'visual' for normal UI, 'text' for ASCII (debugging/testing) */
  renderMode?: RenderMode
  /** Optional version number for debugging */
  version?: number
  /** Whether state is optimistic (unconfirmed) */
  isOptimistic?: boolean
}

/**
 * Text-based board renderer for debugging and Playwright testing.
 * Provides deterministic ASCII output with rich data attributes.
 */
function TextBoardRenderer({
  board,
  currentPlayer,
  winner,
  onColumnClick,
  disabled = false,
  version,
  isOptimistic,
}: Pick<GameBoardProps, 'board' | 'currentPlayer' | 'winner' | 'onColumnClick' | 'disabled' | 'version' | 'isOptimistic'>) {
  const winningCells = winner && winner !== 'draw' ? getWinningCells(board) : null
  const winningSet = new Set(winningCells?.map(([r, c]) => `${r},${c}`) ?? [])
  const isInteractive = !disabled && winner === null && onColumnClick

  const handleColumnClick = (column: number) => {
    if (isInteractive && onColumnClick) {
      onColumnClick(column)
    }
  }

  // Generate ASCII representation
  const renderCell = (cell: Player | null, row: number, col: number): string => {
    if (cell === null) return '.'
    const isWinning = winningSet.has(`${row},${col}`)
    if (cell === 1) return isWinning ? 'R' : 'r'
    return isWinning ? 'Y' : 'y'
  }

  const winnerDisplay = winner === null ? 'none' : winner === 'draw' ? 'draw' : `P${winner}`

  return (
    <div
      className="font-mono text-sm bg-gray-100 dark:bg-gray-800 p-4 rounded-lg"
      data-testid="text-board"
      data-render-mode="text"
      data-current-player={currentPlayer}
      data-winner={winner ?? 'none'}
      data-version={version ?? 0}
      data-optimistic={isOptimistic ?? false}
    >
      {/* Column buttons for interactivity */}
      {isInteractive && (
        <div className="flex gap-1 mb-2" data-testid="text-columns">
          {Array.from({ length: COLUMNS }, (_, col) => (
            <button
              key={col}
              onClick={() => handleColumnClick(col)}
              className="w-6 h-6 bg-blue-500 text-white rounded hover:bg-blue-600 text-xs"
              data-testid={`text-col-${col}`}
            >
              {col}
            </button>
          ))}
        </div>
      )}

      {/* ASCII board */}
      <pre className="leading-tight" data-testid="ascii-board">
        {board.map((row, rowIndex) => (
          <div key={rowIndex} data-testid={`text-row-${rowIndex}`}>
            |{row.map((cell, colIndex) => {
              const char = renderCell(cell, rowIndex, colIndex)
              return (
                <span
                  key={colIndex}
                  data-testid={`text-cell-${rowIndex}-${colIndex}`}
                  data-cell={cell ?? 'empty'}
                  data-winning={winningSet.has(`${rowIndex},${colIndex}`)}
                >
                  {char}
                </span>
              )
            }).reduce((prev, curr) => (
              <>{prev}|{curr}</>
            ))}|
          </div>
        ))}
        <div>+{'-'.repeat(COLUMNS * 2 - 1)}+</div>
        <div> {Array.from({ length: COLUMNS }, (_, i) => i).join(' ')} </div>
      </pre>

      {/* Game info */}
      <div
        className="mt-2 text-xs text-gray-600 dark:text-gray-400 space-y-1"
        data-testid="text-game-info"
      >
        <div>
          Turn: <span data-testid="text-turn">{currentPlayer === 1 ? 'R' : 'Y'}</span> |
          Winner: <span data-testid="text-winner">{winnerDisplay}</span>
        </div>
        {version !== undefined && (
          <div>
            Version: <span data-testid="text-version">{version}</span>
            {isOptimistic && <span className="ml-2 text-yellow-600" data-testid="text-optimistic">(optimistic)</span>}
          </div>
        )}
      </div>
    </div>
  )
}

/**
 * Renders the MakeFour game board.
 * Supports interactive play and read-only replay mode.
 * Note: Animations are currently disabled for rendering stability.
 */
export default function GameBoard({
  board,
  currentPlayer,
  winner,
  onColumnClick,
  disabled = false,
  threats = [],
  showThreats = false,
  renderMode = 'visual',
  version,
  isOptimistic,
}: GameBoardProps) {
  // Use text renderer if requested
  if (renderMode === 'text') {
    return (
      <TextBoardRenderer
        board={board}
        currentPlayer={currentPlayer}
        winner={winner}
        onColumnClick={onColumnClick}
        disabled={disabled}
        version={version}
        isOptimistic={isOptimistic}
      />
    )
  }
  const winningCells = winner && winner !== 'draw' ? getWinningCells(board) : null
  const winningSet = new Set(winningCells?.map(([r, c]) => `${r},${c}`) ?? [])

  // Create threat lookup sets for highlighting
  const winThreatCells = new Set(
    showThreats
      ? threats
          .filter((t) => t.type === 'win')
          .map((t) => `${t.row},${t.column}`)
      : []
  )
  const blockThreatCells = new Set(
    showThreats
      ? threats
          .filter((t) => t.type === 'block')
          .map((t) => `${t.row},${t.column}`)
      : []
  )
  const threatColumns = new Set(
    showThreats ? threats.map((t) => t.column) : []
  )

  const isInteractive = !disabled && winner === null && onColumnClick

  const handleColumnClick = (column: number) => {
    if (isInteractive) {
      onColumnClick(column)
    }
  }

  return (
    <div className="flex flex-col items-center gap-2 sm:gap-4 w-full max-w-[calc(100vw-2rem)] sm:max-w-none">
      {/* Column click targets - invisible buttons above each column */}
      <div className="flex gap-0.5 sm:gap-1">
        {Array.from({ length: COLUMNS }, (_, col) => {
          const hasThreat = threatColumns.has(col)
          const isWinColumn = showThreats && threats.some((t) => t.column === col && t.type === 'win')
          const isBlockColumn = showThreats && threats.some((t) => t.column === col && t.type === 'block')

          return (
            <button
              key={`col-${col}`}
              onClick={() => handleColumnClick(col)}
              disabled={!isInteractive}
              className={cn(
                'w-11 h-11 sm:w-14 sm:h-12 rounded-t-lg transition-colors relative',
                'touch-manipulation', // Prevents double-tap zoom on mobile
                isInteractive
                  ? 'hover:bg-gray-200 dark:hover:bg-gray-700 active:bg-gray-300 dark:active:bg-gray-600 cursor-pointer'
                  : 'cursor-default',
                // Threat highlighting for column headers
                hasThreat && isWinColumn && 'bg-green-100 dark:bg-green-900/30',
                hasThreat && isBlockColumn && !isWinColumn && 'bg-red-100 dark:bg-red-900/30'
              )}
              aria-label={`Drop piece in column ${col + 1}`}
            >
              {/* Threat indicator dot */}
              {hasThreat && (
                <div
                  className={cn(
                    'absolute top-1 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full',
                    isWinColumn ? 'bg-green-500' : 'bg-red-500'
                  )}
                />
              )}
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
          )
        })}
      </div>

      {/* Game board */}
      <div className="bg-blue-600 dark:bg-blue-800 p-1.5 sm:p-2 rounded-lg shadow-lg">
        <div className="grid gap-0.5 sm:gap-1" style={{ gridTemplateColumns: `repeat(${COLUMNS}, 1fr)` }}>
          {board.map((row, rowIndex) =>
            row.map((cell, colIndex) => {
              const isWinningCell = winningSet.has(`${rowIndex},${colIndex}`)
              const isWinThreat = winThreatCells.has(`${rowIndex},${colIndex}`)
              const isBlockThreat = blockThreatCells.has(`${rowIndex},${colIndex}`)
              const isThreatCell = isWinThreat || isBlockThreat

              return (
                <button
                  key={`${rowIndex}-${colIndex}`}
                  onClick={() => handleColumnClick(colIndex)}
                  disabled={!isInteractive}
                  className={cn(
                    'w-11 h-11 sm:w-14 sm:h-14 rounded-full transition-all duration-200',
                    'flex items-center justify-center',
                    'touch-manipulation', // Prevents double-tap zoom on mobile
                    isInteractive ? 'cursor-pointer active:scale-95' : 'cursor-default',
                    // Cell background (the "hole" in the board)
                    'bg-gray-100 dark:bg-gray-900',
                    // Threat highlighting for empty cells where piece would land
                    isThreatCell && cell === null && 'ring-2 ring-inset',
                    isWinThreat && cell === null && 'ring-green-500',
                    isBlockThreat && !isWinThreat && cell === null && 'ring-red-500'
                  )}
                  aria-label={
                    cell === null
                      ? `Empty cell, column ${colIndex + 1}${isThreatCell ? ' (threat)' : ''}`
                      : `Player ${cell} piece, column ${colIndex + 1}`
                  }
                >
                  {/* Threat indicator for empty cells */}
                  {cell === null && isThreatCell && (
                    <div
                      className={cn(
                        'w-9 h-9 sm:w-12 sm:h-12 rounded-full border-2 border-dashed',
                        isWinThreat ? 'border-green-500' : 'border-red-500'
                      )}
                    />
                  )}
                  {/* Piece */}
                  {cell !== null && (
                    <div
                      className={cn(
                        'w-9 h-9 sm:w-12 sm:h-12 rounded-full',
                        cell === 1
                          ? 'bg-red-500 shadow-red-600/50'
                          : 'bg-yellow-400 shadow-yellow-500/50',
                        'shadow-lg',
                        // Highlight winning pieces with a static ring (animations disabled for stability)
                        isWinningCell && 'ring-4 ring-white'
                      )}
                    />
                  )}
                </button>
              )
            })
          )}
        </div>
      </div>

    </div>
  )
}
