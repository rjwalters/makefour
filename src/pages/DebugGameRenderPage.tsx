/**
 * Debug Game Render Page - Test game rendering in isolation
 *
 * Features:
 * - Both players controlled by the same user (click to play either side)
 * - Comprehensive logging of:
 *   - Board state changes
 *   - React render cycles
 *   - DOM measurements
 *   - Animation frames
 *   - Click/touch events
 *   - Piece drop animations
 *   - Win detection rendering
 * - No server required - runs entirely locally
 * - Data attributes for headless browser testing
 */

import { useState, useCallback, useEffect, useRef, useMemo } from 'react'
import { Button } from '../components/ui/button'
import { cn } from '@/lib/utils'
import GameBoard, { type RenderMode } from '../components/GameBoard'
import { useGameState } from '../hooks/useGameState'
import {
  createGameState,
  makeMove,
  replayMoves,
  getWinningCells,
  getValidMoves,
  COLUMNS,
  type GameState,
  type Board,
  type Player,
  type GameResult,
} from '../game/makefour'

interface RenderLog {
  timestamp: number
  type: 'render' | 'state' | 'click' | 'animation' | 'measure' | 'win' | 'move' | 'error'
  message: string
  data?: Record<string, unknown>
}

// Custom GameBoard with logging
interface LoggingGameBoardProps {
  board: Board
  currentPlayer: Player
  winner: GameResult
  onColumnClick: (column: number) => void
  disabled?: boolean
  onLog: (type: RenderLog['type'], message: string, data?: Record<string, unknown>) => void
  renderCount: number
}

function LoggingGameBoard({
  board,
  currentPlayer,
  winner,
  onColumnClick,
  disabled = false,
  onLog,
  renderCount,
}: LoggingGameBoardProps) {
  const boardRef = useRef<HTMLDivElement>(null)
  const lastRenderRef = useRef<number>(0)
  const animationFrameRef = useRef<number | null>(null)
  const winningCells = winner && winner !== 'draw' ? getWinningCells(board) : null
  const winningSet = new Set(winningCells?.map(([r, c]) => `${r},${c}`) ?? [])
  const isInteractive = !disabled && winner === null

  // Log render timing
  useEffect(() => {
    const now = performance.now()
    const delta = lastRenderRef.current ? now - lastRenderRef.current : 0
    onLog('render', `GameBoard render #${renderCount}`, {
      deltaMs: delta.toFixed(2),
      currentPlayer,
      winner,
      pieceCount: board.flat().filter(c => c !== null).length,
    })
    lastRenderRef.current = now
  })

  // Log DOM measurements after render
  useEffect(() => {
    if (boardRef.current) {
      const rect = boardRef.current.getBoundingClientRect()
      onLog('measure', 'Board DOM measurements', {
        width: rect.width.toFixed(0),
        height: rect.height.toFixed(0),
        top: rect.top.toFixed(0),
        left: rect.left.toFixed(0),
      })
    }
  }, [board, onLog])

  // Track animation frames when winning
  useEffect(() => {
    if (winningCells) {
      onLog('win', 'Win detected - starting highlight', {
        cells: winningCells,
        winner,
      })

      let frameCount = 0
      const startTime = performance.now()

      const trackFrame = () => {
        frameCount++
        const elapsed = performance.now() - startTime
        if (elapsed < 1000) {
          // Track for 1 second
          animationFrameRef.current = requestAnimationFrame(trackFrame)
        } else {
          onLog('animation', 'Win animation frame tracking complete', {
            frameCount,
            avgFps: (frameCount / (elapsed / 1000)).toFixed(1),
            totalMs: elapsed.toFixed(0),
          })
        }
      }

      animationFrameRef.current = requestAnimationFrame(trackFrame)

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current)
        }
      }
    }
  }, [winningCells, winner, onLog])

  const handleColumnClick = (column: number, event: React.MouseEvent | React.TouchEvent) => {
    const eventType = 'touches' in event ? 'touch' : 'click'
    const target = event.currentTarget as HTMLElement
    const rect = target.getBoundingClientRect()

    onLog('click', `Column ${column} ${eventType}`, {
      column,
      eventType,
      isInteractive,
      disabled,
      winner,
      targetRect: {
        width: rect.width.toFixed(0),
        height: rect.height.toFixed(0),
      },
      clientX: 'clientX' in event ? event.clientX : (event as React.TouchEvent).touches[0]?.clientX,
      clientY: 'clientY' in event ? event.clientY : (event as React.TouchEvent).touches[0]?.clientY,
    })

    if (isInteractive) {
      onColumnClick(column)
    }
  }

  return (
    <div
      ref={boardRef}
      className="flex flex-col items-center gap-2 w-full"
      data-testid="game-board"
      data-render-count={renderCount}
      data-current-player={currentPlayer}
      data-winner={winner || 'none'}
    >
      {/* Column click targets */}
      <div className="flex gap-1">
        {Array.from({ length: COLUMNS }, (_, col) => (
          <button
            key={`col-${col}`}
            onClick={(e) => handleColumnClick(col, e)}
            onTouchStart={(e) => handleColumnClick(col, e)}
            disabled={!isInteractive}
            className={cn(
              'w-12 h-10 rounded-t-lg transition-colors',
              isInteractive
                ? 'hover:bg-gray-200 dark:hover:bg-gray-700 cursor-pointer'
                : 'cursor-default'
            )}
            data-testid={`column-${col}`}
            data-column={col}
          >
            {isInteractive && (
              <div
                className={cn(
                  'w-8 h-8 mx-auto rounded-full opacity-0 hover:opacity-50 transition-opacity',
                  currentPlayer === 1 ? 'bg-red-500' : 'bg-yellow-500'
                )}
              />
            )}
          </button>
        ))}
      </div>

      {/* Game board */}
      <div className="bg-blue-600 dark:bg-blue-800 p-2 rounded-lg shadow-lg">
        <div
          className="grid gap-1"
          style={{ gridTemplateColumns: `repeat(${COLUMNS}, 1fr)` }}
          data-testid="board-grid"
        >
          {board.map((row, rowIndex) =>
            row.map((cell, colIndex) => {
              const isWinningCell = winningSet.has(`${rowIndex},${colIndex}`)
              const cellId = `cell-${rowIndex}-${colIndex}`

              return (
                <button
                  key={cellId}
                  onClick={(e) => handleColumnClick(colIndex, e)}
                  disabled={!isInteractive}
                  className={cn(
                    'w-12 h-12 rounded-full transition-all duration-200',
                    'flex items-center justify-center',
                    isInteractive ? 'cursor-pointer active:scale-95' : 'cursor-default',
                    'bg-gray-100 dark:bg-gray-900'
                  )}
                  data-testid={cellId}
                  data-row={rowIndex}
                  data-col={colIndex}
                  data-cell={cell || 'empty'}
                  data-winning={isWinningCell}
                >
                  {cell !== null && (
                    <div
                      className={cn(
                        'w-10 h-10 rounded-full shadow-lg',
                        'transition-all duration-300',
                        cell === 1 ? 'bg-red-500' : 'bg-yellow-400',
                        isWinningCell && 'ring-4 ring-white animate-pulse'
                      )}
                      data-testid={`piece-${rowIndex}-${colIndex}`}
                      data-player={cell}
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

export default function DebugGameRenderPage() {
  const [gameState, setGameState] = useState<GameState>(createGameState)
  const [logs, setLogs] = useState<RenderLog[]>([])
  const [renderCount, setRenderCount] = useState(0)
  const [autoPlay, setAutoPlay] = useState(false)
  const [autoPlayInterval, setAutoPlayInterval] = useState(500)
  const [renderMode, setRenderMode] = useState<RenderMode>('visual')
  const [useNewStateHook, setUseNewStateHook] = useState(false)
  const logTextareaRef = useRef<HTMLTextAreaElement>(null)
  const autoPlayRef = useRef<NodeJS.Timeout | null>(null)
  const pageRenderCount = useRef(0)

  // New useGameState hook for comparison
  const newGameState = useGameState()

  // Track page renders
  pageRenderCount.current++

  const addLog = useCallback(
    (type: RenderLog['type'], message: string, data?: Record<string, unknown>) => {
      const log: RenderLog = {
        timestamp: performance.now(),
        type,
        message,
        data,
      }
      setLogs((prev) => [...prev.slice(-500), log]) // Keep last 500 logs
    },
    []
  )

  // Log page state changes
  useEffect(() => {
    addLog('state', 'Game state updated', {
      moveCount: gameState.moveHistory.length,
      currentPlayer: gameState.currentPlayer,
      winner: gameState.winner,
      validMoves: getValidMoves(gameState.board),
    })
    setRenderCount((c) => c + 1)
  }, [gameState, addLog])

  // Auto-scroll log textarea
  useEffect(() => {
    if (logTextareaRef.current) {
      logTextareaRef.current.scrollTop = logTextareaRef.current.scrollHeight
    }
  }, [logs])

  // Auto-play logic
  useEffect(() => {
    if (autoPlay && !gameState.winner) {
      const validMoves = getValidMoves(gameState.board)
      if (validMoves.length > 0) {
        autoPlayRef.current = setTimeout(() => {
          const randomColumn = validMoves[Math.floor(Math.random() * validMoves.length)]
          addLog('move', `Auto-play: Player ${gameState.currentPlayer} plays column ${randomColumn}`, {
            column: randomColumn,
            player: gameState.currentPlayer,
          })
          handleMove(randomColumn)
        }, autoPlayInterval)
      }
    }

    return () => {
      if (autoPlayRef.current) {
        clearTimeout(autoPlayRef.current)
      }
    }
  }, [autoPlay, gameState, autoPlayInterval])

  const handleMove = useCallback(
    (column: number) => {
      const startTime = performance.now()

      addLog('move', `Player ${gameState.currentPlayer} attempting move`, {
        column,
        player: gameState.currentPlayer,
        moveNumber: gameState.moveHistory.length + 1,
      })

      const newState = makeMove(gameState, column)

      if (newState) {
        const elapsed = performance.now() - startTime

        addLog('state', 'Move applied successfully', {
          column,
          player: gameState.currentPlayer,
          newCurrentPlayer: newState.currentPlayer,
          winner: newState.winner,
          processingMs: elapsed.toFixed(2),
          moveHistory: newState.moveHistory,
        })

        setGameState(newState)
      } else {
        addLog('error', 'Invalid move rejected', {
          column,
          validMoves: getValidMoves(gameState.board),
        })
      }
    },
    [gameState, addLog]
  )

  const handleReset = useCallback(() => {
    addLog('state', 'Game reset')
    setGameState(createGameState())
    newGameState.reset()
    setAutoPlay(false)
  }, [addLog, newGameState])

  const handleUndo = useCallback(() => {
    if (gameState.moveHistory.length > 0) {
      const newMoves = gameState.moveHistory.slice(0, -1)
      const newState = replayMoves(newMoves)
      if (newState) {
        addLog('state', 'Undo last move', {
          removedMove: gameState.moveHistory[gameState.moveHistory.length - 1],
          newMoveCount: newMoves.length,
        })
        setGameState(newState)
      }
    }
  }, [gameState, addLog])

  const toggleAutoPlay = useCallback(() => {
    setAutoPlay((prev) => {
      addLog('state', prev ? 'Auto-play stopped' : 'Auto-play started', {
        interval: autoPlayInterval,
      })
      return !prev
    })
  }, [autoPlayInterval, addLog])

  const clearLogs = useCallback(() => {
    setLogs([])
    addLog('state', 'Logs cleared')
  }, [addLog])

  // Format logs for display
  const formattedLogs = useMemo(() => {
    return logs
      .map((log) => {
        const time = log.timestamp.toFixed(2).padStart(10, ' ')
        const type = log.type.toUpperCase().padEnd(9, ' ')
        let line = `[${time}ms] [${type}] ${log.message}`
        if (log.data) {
          line += `\n           DATA: ${JSON.stringify(log.data)}`
        }
        return line
      })
      .join('\n')
  }, [logs])

  return (
    <div
      className="min-h-screen bg-gray-50 dark:bg-gray-900 p-4"
      data-testid="debug-game-render-page"
      data-page-renders={pageRenderCount.current}
    >
      <div className="max-w-6xl mx-auto">
        <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-white">
          Debug: Game Rendering
        </h1>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Game Board Section */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Game Board
              </h2>
              <div className="flex items-center gap-4">
                <div className="text-sm text-gray-500" data-testid="render-count">
                  Renders: {renderCount}
                </div>
                {/* Render mode toggle */}
                <div className="flex items-center gap-2">
                  <label className="text-xs text-gray-500">Mode:</label>
                  <select
                    value={renderMode}
                    onChange={(e) => setRenderMode(e.target.value as RenderMode)}
                    className="text-xs p-1 rounded border dark:bg-gray-700"
                    data-testid="render-mode-select"
                  >
                    <option value="visual">Visual</option>
                    <option value="text">Text/ASCII</option>
                  </select>
                </div>
                {/* Hook toggle */}
                <div className="flex items-center gap-1">
                  <input
                    type="checkbox"
                    id="use-new-hook"
                    checked={useNewStateHook}
                    onChange={(e) => setUseNewStateHook(e.target.checked)}
                    className="w-3 h-3"
                    data-testid="use-new-hook-checkbox"
                  />
                  <label htmlFor="use-new-hook" className="text-xs text-gray-500">
                    useGameState
                  </label>
                </div>
              </div>
            </div>

            {/* Game status */}
            <div
              className="mb-4 p-3 rounded bg-gray-100 dark:bg-gray-700 text-sm"
              data-testid="game-status"
            >
              <div className="grid grid-cols-2 gap-2">
                <div>
                  Current Player:{' '}
                  <span
                    className={cn(
                      'font-bold',
                      gameState.currentPlayer === 1 ? 'text-red-500' : 'text-yellow-500'
                    )}
                    data-testid="current-player"
                  >
                    {gameState.currentPlayer === 1 ? 'Red (1)' : 'Yellow (2)'}
                  </span>
                </div>
                <div>
                  Moves: <span data-testid="move-count">{gameState.moveHistory.length}</span>
                </div>
                <div>
                  Winner:{' '}
                  <span data-testid="winner">
                    {gameState.winner
                      ? gameState.winner === 'draw'
                        ? 'Draw'
                        : `Player ${gameState.winner}`
                      : 'None'}
                  </span>
                </div>
                <div>
                  Valid Moves:{' '}
                  <span data-testid="valid-moves">
                    {getValidMoves(gameState.board).join(', ') || 'None'}
                  </span>
                </div>
              </div>
            </div>

            {/* Board */}
            <div className="flex justify-center mb-4">
              {renderMode === 'text' ? (
                <GameBoard
                  board={useNewStateHook ? newGameState.board : gameState.board}
                  currentPlayer={useNewStateHook ? newGameState.currentPlayer : gameState.currentPlayer}
                  winner={useNewStateHook ? newGameState.winner : gameState.winner}
                  onColumnClick={useNewStateHook ? (col) => {
                    addLog('move', `New hook: applying move to column ${col}`)
                    newGameState.applyMove(col)
                  } : handleMove}
                  renderMode="text"
                  version={useNewStateHook ? newGameState.version : gameState.moveHistory.length}
                  isOptimistic={useNewStateHook ? newGameState.isOptimistic : false}
                />
              ) : (
                <LoggingGameBoard
                  board={useNewStateHook ? newGameState.board : gameState.board}
                  currentPlayer={useNewStateHook ? newGameState.currentPlayer : gameState.currentPlayer}
                  winner={useNewStateHook ? newGameState.winner : gameState.winner}
                  onColumnClick={useNewStateHook ? (col) => {
                    addLog('move', `New hook: applying move to column ${col}`)
                    newGameState.applyMove(col)
                  } : handleMove}
                  onLog={addLog}
                  renderCount={renderCount}
                />
              )}
            </div>

            {/* Version info when using new hook */}
            {useNewStateHook && (
              <div className="mb-4 p-2 rounded bg-blue-50 dark:bg-blue-900/20 text-xs" data-testid="version-info">
                <div className="font-medium text-blue-700 dark:text-blue-300">useGameState Debug Info</div>
                <div className="mt-1 grid grid-cols-2 gap-2 text-blue-600 dark:text-blue-400">
                  <div>Version: <span data-testid="state-version">{newGameState.version}</span></div>
                  <div>Optimistic: <span data-testid="state-optimistic">{String(newGameState.isOptimistic)}</span></div>
                  <div>Moves: <span data-testid="state-moves">{newGameState.moves.length}</span></div>
                  <div>Winner: <span data-testid="state-winner">{newGameState.winner ?? 'none'}</span></div>
                </div>
              </div>
            )}

            {/* Controls */}
            <div className="flex flex-wrap gap-2">
              <Button onClick={handleReset} variant="outline" data-testid="reset-btn">
                Reset Game
              </Button>
              <Button
                onClick={handleUndo}
                variant="outline"
                disabled={gameState.moveHistory.length === 0}
                data-testid="undo-btn"
              >
                Undo Move
              </Button>
              <Button
                onClick={toggleAutoPlay}
                variant={autoPlay ? 'destructive' : 'default'}
                disabled={!!gameState.winner}
                data-testid="autoplay-btn"
              >
                {autoPlay ? 'Stop Auto-Play' : 'Start Auto-Play'}
              </Button>
            </div>

            {/* Auto-play speed control */}
            <div className="mt-4 flex items-center gap-2">
              <label className="text-sm text-gray-600 dark:text-gray-400">
                Auto-play interval (ms):
              </label>
              <input
                type="range"
                min="100"
                max="2000"
                step="100"
                value={autoPlayInterval}
                onChange={(e) => setAutoPlayInterval(parseInt(e.target.value))}
                className="w-32"
                data-testid="autoplay-interval"
              />
              <span className="text-sm text-gray-500" data-testid="autoplay-interval-value">
                {autoPlayInterval}ms
              </span>
            </div>

            {/* Move history */}
            <div className="mt-4">
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Move History
              </h3>
              <div
                className="p-2 bg-gray-100 dark:bg-gray-700 rounded text-sm font-mono"
                data-testid="move-history"
              >
                {gameState.moveHistory.length > 0
                  ? gameState.moveHistory.map((m, i) => (
                      <span key={i} className={i % 2 === 0 ? 'text-red-500' : 'text-yellow-500'}>
                        {m}
                        {i < gameState.moveHistory.length - 1 ? ', ' : ''}
                      </span>
                    ))
                  : 'No moves yet'}
              </div>
            </div>
          </div>

          {/* Debug Log Section */}
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Debug Log</h2>
              <Button onClick={clearLogs} variant="outline" size="sm" data-testid="clear-logs-btn">
                Clear
              </Button>
            </div>

            <textarea
              ref={logTextareaRef}
              readOnly
              value={formattedLogs}
              className="w-full h-[500px] p-2 font-mono text-xs bg-gray-900 text-green-400 rounded border-0 resize-none"
              data-testid="debug-log"
              id="debug-log"
            />

            <div className="mt-2 text-xs text-gray-500">
              Log types: RENDER (component renders), STATE (game state changes), CLICK (user
              interactions), ANIMATION (frame tracking), MEASURE (DOM measurements), WIN (win
              detection), MOVE (move processing), ERROR (failures)
            </div>
          </div>
        </div>

        {/* Board state JSON */}
        <div className="mt-6 bg-white dark:bg-gray-800 rounded-lg p-6 shadow">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            Board State (JSON)
          </h2>
          <pre
            className="p-4 bg-gray-100 dark:bg-gray-700 rounded text-xs font-mono overflow-x-auto"
            data-testid="board-json"
          >
            {JSON.stringify(
              {
                board: gameState.board,
                currentPlayer: gameState.currentPlayer,
                winner: gameState.winner,
                moveHistory: gameState.moveHistory,
              },
              null,
              2
            )}
          </pre>
        </div>
      </div>
    </div>
  )
}
