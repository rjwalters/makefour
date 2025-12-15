import { useState, useEffect, useCallback, useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import GameBoard from '../components/GameBoard'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { getStateAtMove, type GameState } from '../game/makefour'
import {
  analyzeGame,
  summarizeGameQuality,
  getMoveQualityColor,
  getMoveQualityLabel,
  type MoveAnalysis,
  type GameQualitySummary,
} from '../ai/moveQuality'

interface GameData {
  id: string
  outcome: 'win' | 'loss' | 'draw'
  moves: number[]
  moveCount: number
  createdAt: number
}

export default function ReplayPage() {
  const { logout, user } = useAuth()
  const { gameId } = useParams<{ gameId: string }>()
  const { apiCall } = useAuthenticatedApi()

  const [game, setGame] = useState<GameData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Current move index for replay (0 = empty board, moves.length = final state)
  const [currentMoveIndex, setCurrentMoveIndex] = useState(0)
  const [gameState, setGameState] = useState<GameState | null>(null)

  // Move quality analysis
  const [moveAnalyses, setMoveAnalyses] = useState<MoveAnalysis[]>([])
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  // Compute summary from analyses
  const qualitySummary = useMemo<GameQualitySummary | null>(() => {
    if (moveAnalyses.length === 0) return null
    return summarizeGameQuality(moveAnalyses)
  }, [moveAnalyses])

  // Get current move analysis (for the move that led to current state)
  const currentMoveAnalysis = useMemo<MoveAnalysis | null>(() => {
    if (currentMoveIndex === 0 || moveAnalyses.length === 0) return null
    return moveAnalyses[currentMoveIndex - 1] || null
  }, [currentMoveIndex, moveAnalyses])

  // Fetch game data
  useEffect(() => {
    const fetchGame = async () => {
      if (!gameId) return

      setIsLoading(true)
      setError(null)

      try {
        const data = await apiCall<GameData>(`/api/games/${gameId}`)
        setGame(data)
        // Start at final position
        setCurrentMoveIndex(data.moves.length)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load game')
      } finally {
        setIsLoading(false)
      }
    }

    fetchGame()
  }, [gameId])

  // Analyze game moves for quality scoring
  useEffect(() => {
    const performAnalysis = async () => {
      if (!game || game.moves.length === 0) return

      setIsAnalyzing(true)
      try {
        const analyses = await analyzeGame(game.moves)
        setMoveAnalyses(analyses)
      } catch (err) {
        console.error('Failed to analyze game:', err)
      } finally {
        setIsAnalyzing(false)
      }
    }

    performAnalysis()
  }, [game])

  // Update game state when move index changes
  useEffect(() => {
    if (!game) return

    const state = getStateAtMove(game.moves, currentMoveIndex)
    setGameState(state)
  }, [game, currentMoveIndex])

  const goToStart = useCallback(() => {
    setCurrentMoveIndex(0)
  }, [])

  const goBack = useCallback(() => {
    setCurrentMoveIndex((prev) => Math.max(0, prev - 1))
  }, [])

  const goForward = useCallback(() => {
    if (!game) return
    setCurrentMoveIndex((prev) => Math.min(game.moves.length, prev + 1))
  }, [game])

  const goToEnd = useCallback(() => {
    if (!game) return
    setCurrentMoveIndex(game.moves.length)
  }, [game])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'ArrowLeft':
          goBack()
          break
        case 'ArrowRight':
          goForward()
          break
        case 'Home':
          goToStart()
          break
        case 'End':
          goToEnd()
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [goBack, goForward, goToStart, goToEnd])

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getOutcomeLabel = (outcome: GameData['outcome']) => {
    switch (outcome) {
      case 'win':
        return 'You won!'
      case 'loss':
        return 'You lost'
      case 'draw':
        return 'Draw'
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
        <p className="text-lg text-muted-foreground">Loading game...</p>
      </div>
    )
  }

  if (error || !game || !gameState) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center">
            <p className="text-red-500 mb-4">{error || 'Game not found'}</p>
            <Link to="/games">
              <Button>Back to Games</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <Link to="/dashboard" className="text-2xl font-bold hover:opacity-80">
              MakeFour
            </Link>
            {user && (
              <p className="text-xs text-muted-foreground">{user.email}</p>
            )}
          </div>
          <div className="flex gap-2">
            <Link to="/games">
              <Button variant="outline" size="sm">
                My Games
              </Button>
            </Link>
            <Link to="/play">
              <Button size="sm">Play</Button>
            </Link>
            <ThemeToggle />
            <Button variant="outline" onClick={logout} size="sm">
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-lg mx-auto">
          <Card>
            <CardHeader className="text-center">
              <CardTitle className="text-2xl">Game Replay</CardTitle>
              <p className="text-muted-foreground">{formatDate(game.createdAt)}</p>
              <p className="text-lg font-semibold mt-2">{getOutcomeLabel(game.outcome)}</p>
            </CardHeader>
            <CardContent className="flex flex-col items-center gap-6">
              <GameBoard
                board={gameState.board}
                currentPlayer={gameState.currentPlayer}
                winner={gameState.winner}
                disabled={true}
              />

              {/* Replay controls */}
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goToStart}
                  disabled={currentMoveIndex === 0}
                  title="Go to start (Home)"
                >
                  ⏮
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goBack}
                  disabled={currentMoveIndex === 0}
                  title="Previous move (←)"
                >
                  ◀
                </Button>
                <span className="px-4 text-sm font-medium min-w-[80px] text-center">
                  {currentMoveIndex} / {game.moves.length}
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goForward}
                  disabled={currentMoveIndex === game.moves.length}
                  title="Next move (→)"
                >
                  ▶
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={goToEnd}
                  disabled={currentMoveIndex === game.moves.length}
                  title="Go to end (End)"
                >
                  ⏭
                </Button>
              </div>

              <p className="text-xs text-muted-foreground">
                Use arrow keys to navigate
              </p>

              {/* Current move quality info */}
              {currentMoveAnalysis && (
                <div className="w-full p-3 rounded-lg bg-muted/50">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium ${getMoveQualityColor(currentMoveAnalysis.quality)}`}
                      >
                        {getMoveQualityLabel(currentMoveAnalysis.quality)}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        Move {currentMoveIndex}: Column {currentMoveAnalysis.column + 1}
                      </span>
                    </div>
                    {currentMoveAnalysis.quality !== 'optimal' && (
                      <span className="text-xs text-muted-foreground">
                        Best: Column {currentMoveAnalysis.optimalMove + 1}
                      </span>
                    )}
                  </div>
                  {(currentMoveAnalysis.quality === 'mistake' ||
                    currentMoveAnalysis.quality === 'blunder') && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {currentMoveAnalysis.quality === 'blunder'
                        ? 'This move significantly worsened the position.'
                        : 'A better move was available.'}
                    </p>
                  )}
                </div>
              )}

              {/* Move list with quality colors */}
              <div className="w-full">
                <p className="text-sm font-medium mb-2">
                  Moves {isAnalyzing && <span className="text-xs text-muted-foreground">(analyzing...)</span>}
                </p>
                <div className="flex flex-wrap gap-1">
                  {game.moves.map((col, idx) => {
                    const analysis = moveAnalyses[idx]
                    const isSelected = idx + 1 === currentMoveIndex
                    const qualityClass = analysis
                      ? getMoveQualityColor(analysis.quality)
                      : ''

                    return (
                      <button
                        key={idx}
                        onClick={() => setCurrentMoveIndex(idx + 1)}
                        className={`w-8 h-8 text-xs rounded border transition-all ${
                          isSelected
                            ? 'ring-2 ring-primary ring-offset-2 dark:ring-offset-gray-900'
                            : ''
                        } ${
                          analysis
                            ? qualityClass
                            : idx + 1 < currentMoveIndex
                            ? 'bg-muted'
                            : 'bg-background hover:bg-accent'
                        }`}
                        title={`Move ${idx + 1}: Column ${col + 1}${analysis ? ` (${getMoveQualityLabel(analysis.quality)})` : ''}`}
                      >
                        {col + 1}
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Quality summary statistics */}
              {qualitySummary && (
                <div className="w-full p-4 rounded-lg bg-muted/30 border">
                  <p className="text-sm font-medium mb-3">Move Quality Summary</p>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground text-xs">Optimal moves</p>
                      <p className="font-medium text-green-600 dark:text-green-400">
                        {qualitySummary.counts.optimal} ({qualitySummary.optimalPercentage.toFixed(0)}%)
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground text-xs">Good or better</p>
                      <p className="font-medium">
                        {qualitySummary.counts.optimal + qualitySummary.counts.good} (
                        {qualitySummary.goodOrBetterPercentage.toFixed(0)}%)
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground text-xs">Mistakes</p>
                      <p className="font-medium text-orange-500">
                        {qualitySummary.mistakeCount}
                      </p>
                    </div>
                    <div>
                      <p className="text-muted-foreground text-xs">Blunders</p>
                      <p className="font-medium text-red-500">
                        {qualitySummary.blunderCount}
                      </p>
                    </div>
                  </div>
                  {/* Quality legend */}
                  <div className="mt-4 pt-3 border-t">
                    <p className="text-xs text-muted-foreground mb-2">Legend</p>
                    <div className="flex flex-wrap gap-2">
                      {(['optimal', 'good', 'neutral', 'mistake', 'blunder'] as const).map(
                        (quality) => (
                          <span
                            key={quality}
                            className={`px-2 py-0.5 rounded text-xs ${getMoveQualityColor(quality)}`}
                          >
                            {getMoveQualityLabel(quality)}
                          </span>
                        )
                      )}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
