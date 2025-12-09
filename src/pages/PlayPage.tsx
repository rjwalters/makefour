import { useState, useCallback, useEffect } from 'react'
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
} from '../game/makefour'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { suggestMove } from '../ai/coach'

export default function PlayPage() {
  const { logout, user, isAuthenticated } = useAuth()
  const { apiCall } = useAuthenticatedApi()
  const [gameState, setGameState] = useState<GameState>(createGameState)
  const [isSaving, setIsSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [gameSaved, setGameSaved] = useState(false)
  const [isBotThinking, setIsBotThinking] = useState(false)

  // Player is always Player 1 (red), Bot is Player 2 (yellow)
  const playerNumber = 1
  const botNumber = 2
  const isPlayerTurn = gameState.currentPlayer === playerNumber && gameState.winner === null

  // Bot makes a move when it's the bot's turn
  useEffect(() => {
    if (gameState.currentPlayer !== botNumber || gameState.winner !== null) {
      return
    }

    setIsBotThinking(true)

    const makeBotMove = async () => {
      // Small delay for UX - makes it feel like the bot is "thinking"
      await new Promise((resolve) => setTimeout(resolve, 500))

      const botMove = await suggestMove({
        board: gameState.board,
        currentPlayer: gameState.currentPlayer,
        moveHistory: gameState.moveHistory,
      })

      const newState = makeMove(gameState, botMove)
      if (newState) {
        setGameState(newState)
      }
      setIsBotThinking(false)
    }

    makeBotMove()
  }, [gameState])

  const handleColumnClick = useCallback((column: number) => {
    // Only allow clicks on player's turn
    if (!isPlayerTurn || isBotThinking) return

    const newState = makeMove(gameState, column)
    if (newState) {
      setGameState(newState)
      setGameSaved(false)
      setSaveError(null)
    }
  }, [gameState, isPlayerTurn, isBotThinking])

  const handleNewGame = useCallback(() => {
    setGameState(createGameState())
    setGameSaved(false)
    setSaveError(null)
    setIsBotThinking(false)
  }, [])

  const handleSaveGame = useCallback(async () => {
    if (!gameState.winner || gameSaved) return

    setIsSaving(true)
    setSaveError(null)

    try {
      // Determine outcome from player's perspective (Player 1)
      let outcome: 'win' | 'loss' | 'draw'
      if (gameState.winner === 'draw') {
        outcome = 'draw'
      } else if (gameState.winner === playerNumber) {
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
    if (gameState.winner === playerNumber) {
      return 'You win!'
    }
    if (gameState.winner === botNumber) {
      return 'Bot wins!'
    }
    if (isBotThinking) {
      return 'Bot is thinking...'
    }
    return 'Your turn'
  }

  const getStatusColor = (): string => {
    if (gameState.winner === playerNumber) {
      return 'text-green-600 dark:text-green-400'
    }
    if (gameState.winner === botNumber) {
      return 'text-red-500'
    }
    if (gameState.winner === 'draw') {
      return 'text-muted-foreground'
    }
    if (isBotThinking) {
      return 'text-yellow-500'
    }
    return 'text-red-500' // Player's color
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <Link to="/" className="text-2xl font-bold hover:opacity-80">
              MakeFour
            </Link>
            {user && (
              <p className="text-xs text-muted-foreground">{user.email}</p>
            )}
          </div>
          <div className="flex gap-2">
            {isAuthenticated ? (
              <>
                <Link to="/dashboard">
                  <Button variant="outline" size="sm">
                    Dashboard
                  </Button>
                </Link>
                <ThemeToggle />
                <Button variant="outline" onClick={logout} size="sm">
                  Logout
                </Button>
              </>
            ) : (
              <>
                <ThemeToggle />
                <Link to="/login">
                  <Button variant="outline" size="sm">
                    Login
                  </Button>
                </Link>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-lg mx-auto">
          <Card>
            <CardHeader className="text-center pb-2">
              <div className="flex justify-center items-center gap-4 mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-red-500" />
                  <span className="text-sm font-medium">You</span>
                </div>
                <span className="text-muted-foreground">vs</span>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-yellow-400" />
                  <span className="text-sm font-medium">Bot</span>
                </div>
              </div>
              <CardTitle className={`text-2xl ${getStatusColor()}`}>
                {getStatusMessage()}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col items-center gap-6">
              <GameBoard
                board={gameState.board}
                currentPlayer={gameState.currentPlayer}
                winner={gameState.winner}
                onColumnClick={handleColumnClick}
                disabled={!isPlayerTurn || isBotThinking}
              />

              <div className="flex flex-col gap-3 w-full">
                {/* Game over actions */}
                {gameState.winner !== null && (
                  <div className="flex flex-col gap-2">
                    <div className="flex gap-2 justify-center">
                      <Button onClick={handleNewGame} size="lg">
                        Play Again
                      </Button>
                      {isAuthenticated && !gameSaved && (
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
                    {isAuthenticated && gameSaved && (
                      <p className="text-center text-sm text-green-600 dark:text-green-400">
                        Game saved to history!
                      </p>
                    )}
                    {isAuthenticated && saveError && (
                      <p className="text-center text-sm text-red-600 dark:text-red-400">
                        {saveError}
                      </p>
                    )}
                    {!isAuthenticated && (
                      <p className="text-center text-sm text-muted-foreground">
                        <Link to="/login" className="text-primary hover:underline">
                          Sign in
                        </Link>
                        {' '}to save games and track your progress
                      </p>
                    )}
                  </div>
                )}

                {/* Mid-game actions */}
                {gameState.winner === null && gameState.moveHistory.length > 0 && (
                  <div className="flex gap-2 justify-center">
                    <Button onClick={handleNewGame} variant="outline" size="sm">
                      New Game
                    </Button>
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
