import { useState, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import GameBoard from '../components/GameBoard'
import {
  createGameState,
  makeMove,
  type GameState,
  type Player,
} from '../game/makefour'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { analyzePosition, type Analysis } from '../ai/coach'

export default function PlayPage() {
  const { logout, user } = useAuth()
  const { apiCall } = useAuthenticatedApi()
  const [gameState, setGameState] = useState<GameState>(createGameState)
  const [isSaving, setIsSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [gameSaved, setGameSaved] = useState(false)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleColumnClick = useCallback((column: number) => {
    const newState = makeMove(gameState, column)
    if (newState) {
      setGameState(newState)
      setGameSaved(false)
      setSaveError(null)
      setAnalysis(null) // Clear analysis when move is made
    }
  }, [gameState])

  const handleNewGame = useCallback(() => {
    setGameState(createGameState())
    setGameSaved(false)
    setSaveError(null)
    setAnalysis(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (gameState.winner !== null) return

    setIsAnalyzing(true)
    try {
      const result = await analyzePosition({
        board: gameState.board,
        currentPlayer: gameState.currentPlayer,
        moveHistory: gameState.moveHistory,
      })
      setAnalysis(result)
    } catch (error) {
      console.error('Analysis failed:', error)
    } finally {
      setIsAnalyzing(false)
    }
  }, [gameState])

  const handleSaveGame = useCallback(async () => {
    if (!gameState.winner || gameSaved) return

    setIsSaving(true)
    setSaveError(null)

    try {
      // Determine outcome from Player 1's perspective
      let outcome: 'win' | 'loss' | 'draw'
      if (gameState.winner === 'draw') {
        outcome = 'draw'
      } else if (gameState.winner === 1) {
        outcome = 'win'
      } else {
        outcome = 'loss'
      }

      await apiCall('/api/games', {
        method: 'POST',
        body: JSON.stringify({
          outcome,
          moves: gameState.moveHistory,
        }),
      })

      setGameSaved(true)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save game')
    } finally {
      setIsSaving(false)
    }
  }, [gameState, gameSaved, apiCall])

  const getStatusMessage = (): string => {
    if (gameState.winner === 'draw') {
      return "It's a draw!"
    }
    if (gameState.winner !== null) {
      return `Player ${gameState.winner} wins!`
    }
    return `Player ${gameState.currentPlayer}'s turn`
  }

  const getPlayerColor = (player: Player): string => {
    return player === 1 ? 'text-red-500' : 'text-yellow-500'
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
            <Link to="/dashboard">
              <Button variant="outline" size="sm">
                Dashboard
              </Button>
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
              <CardTitle className="text-3xl">MakeFour</CardTitle>
              <p
                className={`text-lg font-semibold ${
                  gameState.winner !== null && gameState.winner !== 'draw'
                    ? getPlayerColor(gameState.winner as Player)
                    : gameState.winner === null
                    ? getPlayerColor(gameState.currentPlayer)
                    : ''
                }`}
              >
                {getStatusMessage()}
              </p>
            </CardHeader>
            <CardContent className="flex flex-col items-center gap-6">
              <GameBoard
                board={gameState.board}
                currentPlayer={gameState.currentPlayer}
                winner={gameState.winner}
                onColumnClick={handleColumnClick}
              />

              <div className="flex flex-col gap-3 w-full">
                {/* Game over actions */}
                {gameState.winner !== null && (
                  <div className="flex flex-col gap-2">
                    <div className="flex gap-2 justify-center">
                      <Button onClick={handleNewGame} size="lg">
                        Play Again
                      </Button>
                      {!gameSaved && (
                        <Button
                          onClick={handleSaveGame}
                          variant="outline"
                          size="lg"
                          disabled={isSaving}
                        >
                          {isSaving ? 'Saving...' : 'Save Game'}
                        </Button>
                      )}
                    </div>
                    {gameSaved && (
                      <p className="text-center text-sm text-green-600 dark:text-green-400">
                        Game saved to history!
                      </p>
                    )}
                    {saveError && (
                      <p className="text-center text-sm text-red-600 dark:text-red-400">
                        {saveError}
                      </p>
                    )}
                  </div>
                )}

                {/* Mid-game actions */}
                {gameState.winner === null && gameState.moveHistory.length > 0 && (
                  <div className="flex gap-2 justify-center">
                    <Button onClick={handleNewGame} variant="outline">
                      Reset Game
                    </Button>
                    <Button
                      onClick={handleAnalyze}
                      variant="secondary"
                      disabled={isAnalyzing}
                    >
                      {isAnalyzing ? 'Analyzing...' : 'Analyze Position'}
                    </Button>
                  </div>
                )}

                {/* AI Analysis display */}
                {analysis && (
                  <div className="p-3 bg-muted rounded-lg text-sm">
                    <p className="font-medium mb-1">AI Coach (experimental)</p>
                    <p className="text-muted-foreground">{analysis.evaluation}</p>
                    {analysis.bestMove >= 0 && (
                      <p className="mt-1">
                        Suggested move: Column {analysis.bestMove + 1}
                      </p>
                    )}
                  </div>
                )}

                {/* Move counter */}
                <p className="text-center text-sm text-muted-foreground">
                  Moves: {gameState.moveHistory.length}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
