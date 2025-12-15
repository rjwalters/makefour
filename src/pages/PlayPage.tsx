import { useState, useCallback, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import GameBoard from '../components/GameBoard'
import AnalysisPanel from '../components/AnalysisPanel'
import {
  createGameState,
  makeMove,
  type GameState,
  type Player,
} from '../game/makefour'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { suggestMove, analyzeThreats, DIFFICULTY_LEVELS, type DifficultyLevel } from '../ai/coach'

type GameMode = 'ai' | 'hotseat'
type GamePhase = 'setup' | 'playing'

interface GameSettings {
  mode: GameMode
  difficulty: DifficultyLevel
  playerColor: Player // Which player the user plays as (1 = red/first, 2 = yellow/second)
}

const DEFAULT_SETTINGS: GameSettings = {
  mode: 'ai',
  difficulty: 'intermediate',
  playerColor: 1,
}

// Load saved settings from localStorage
function loadSettings(): GameSettings {
  try {
    const saved = localStorage.getItem('makefour-settings')
    if (saved) {
      const parsed = JSON.parse(saved)
      return {
        mode: parsed.mode || DEFAULT_SETTINGS.mode,
        difficulty: parsed.difficulty || DEFAULT_SETTINGS.difficulty,
        playerColor: parsed.playerColor || DEFAULT_SETTINGS.playerColor,
      }
    }
  } catch {
    // Ignore parse errors
  }
  return DEFAULT_SETTINGS
}

// Save settings to localStorage
function saveSettings(settings: GameSettings) {
  localStorage.setItem('makefour-settings', JSON.stringify(settings))
}

export default function PlayPage() {
  const { logout, user, isAuthenticated } = useAuth()
  const { apiCall } = useAuthenticatedApi()
  const [gameState, setGameState] = useState<GameState>(createGameState)
  const [isSaving, setIsSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [gameSaved, setGameSaved] = useState(false)
  const [isBotThinking, setIsBotThinking] = useState(false)
  const [gamePhase, setGamePhase] = useState<GamePhase>('setup')
  const [settings, setSettings] = useState<GameSettings>(loadSettings)
  const [showAnalysis, setShowAnalysis] = useState(false)

  // Determine player numbers based on settings
  const userPlayerNumber = settings.playerColor
  const aiPlayerNumber: Player = settings.playerColor === 1 ? 2 : 1
  const isGameOver = gameState.winner !== null

  // Compute threats for highlighting (memoized for performance)
  const threats = useMemo(() => {
    if (!showAnalysis || isGameOver) return []
    const analysis = analyzeThreats(gameState.board, gameState.currentPlayer)
    return analysis.threats
  }, [gameState.board, gameState.currentPlayer, showAnalysis, isGameOver])

  // AI makes a move when it's the AI's turn in AI mode
  useEffect(() => {
    if (gamePhase !== 'playing') return
    if (settings.mode !== 'ai') return
    if (gameState.currentPlayer !== aiPlayerNumber || gameState.winner !== null) return

    setIsBotThinking(true)

    const makeBotMove = async () => {
      // Small delay for UX - makes it feel like the bot is "thinking"
      await new Promise((resolve) => setTimeout(resolve, 500))

      const botMove = await suggestMove(
        {
          board: gameState.board,
          currentPlayer: gameState.currentPlayer,
          moveHistory: gameState.moveHistory,
        },
        settings.difficulty
      )

      const newState = makeMove(gameState, botMove)
      if (newState) {
        setGameState(newState)
      }
      setIsBotThinking(false)
    }

    makeBotMove()
  }, [gameState, gamePhase, settings.mode, settings.difficulty, aiPlayerNumber])

  const handleColumnClick = useCallback(
    (column: number) => {
      if (gamePhase !== 'playing') return
      if (gameState.winner !== null) return

      // In AI mode, only allow clicks on user's turn
      if (settings.mode === 'ai' && (gameState.currentPlayer !== userPlayerNumber || isBotThinking)) {
        return
      }

      const newState = makeMove(gameState, column)
      if (newState) {
        setGameState(newState)
        setGameSaved(false)
        setSaveError(null)
      }
    },
    [gameState, gamePhase, settings.mode, userPlayerNumber, isBotThinking]
  )

  const handleStartGame = useCallback(() => {
    saveSettings(settings)
    setGameState(createGameState())
    setGameSaved(false)
    setSaveError(null)
    setIsBotThinking(false)
    setGamePhase('playing')
  }, [settings])

  const handleNewGame = useCallback(() => {
    setGamePhase('setup')
    setGameState(createGameState())
    setGameSaved(false)
    setSaveError(null)
    setIsBotThinking(false)
  }, [])

  const handlePlayAgain = useCallback(() => {
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
      // Determine outcome from user's perspective
      let outcome: 'win' | 'loss' | 'draw'
      if (gameState.winner === 'draw') {
        outcome = 'draw'
      } else if (settings.mode === 'hotseat') {
        // In hotseat mode, outcome is from perspective of player 1
        outcome = gameState.winner === 1 ? 'win' : 'loss'
      } else if (gameState.winner === userPlayerNumber) {
        outcome = 'win'
      } else {
        outcome = 'loss'
      }

      await apiCall('/api/games', {
        method: 'POST',
        body: JSON.stringify({
          outcome,
          moves: gameState.moveHistory,
          opponentType: settings.mode === 'ai' ? 'ai' : 'human',
          aiDifficulty: settings.mode === 'ai' ? settings.difficulty : null,
          playerNumber: settings.mode === 'ai' ? userPlayerNumber : 1,
        }),
      })

      setGameSaved(true)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save game')
    } finally {
      setIsSaving(false)
    }
  }, [gameState, gameSaved, apiCall, settings, userPlayerNumber])

  const getStatusMessage = (): string => {
    if (gameState.winner === 'draw') {
      return "It's a draw!"
    }
    if (settings.mode === 'hotseat') {
      if (gameState.winner) {
        return `Player ${gameState.winner} wins!`
      }
      return `Player ${gameState.currentPlayer}'s turn`
    }
    // AI mode
    if (gameState.winner === userPlayerNumber) {
      return 'You win!'
    }
    if (gameState.winner === aiPlayerNumber) {
      return 'AI wins!'
    }
    if (isBotThinking) {
      return 'AI is thinking...'
    }
    return 'Your turn'
  }

  const getStatusColor = (): string => {
    if (gameState.winner === 'draw') {
      return 'text-muted-foreground'
    }
    if (settings.mode === 'hotseat') {
      if (gameState.winner) {
        return gameState.winner === 1 ? 'text-red-500' : 'text-yellow-500'
      }
      return gameState.currentPlayer === 1 ? 'text-red-500' : 'text-yellow-500'
    }
    // AI mode
    if (gameState.winner === userPlayerNumber) {
      return 'text-green-600 dark:text-green-400'
    }
    if (gameState.winner === aiPlayerNumber) {
      return 'text-red-500'
    }
    if (isBotThinking) {
      return 'text-yellow-500'
    }
    return userPlayerNumber === 1 ? 'text-red-500' : 'text-yellow-500'
  }

  const renderSetupScreen = () => (
    <Card>
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">New Game</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Game Mode Selection */}
        <div>
          <label className="block text-sm font-medium mb-2">Game Mode</label>
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant={settings.mode === 'ai' ? 'default' : 'outline'}
              onClick={() => setSettings({ ...settings, mode: 'ai' })}
              className="h-auto py-3"
            >
              <div className="text-center">
                <div className="font-medium">vs AI</div>
                <div className="text-xs opacity-80">Play against the computer</div>
              </div>
            </Button>
            <Button
              variant={settings.mode === 'hotseat' ? 'default' : 'outline'}
              onClick={() => setSettings({ ...settings, mode: 'hotseat' })}
              className="h-auto py-3"
            >
              <div className="text-center">
                <div className="font-medium">Hotseat</div>
                <div className="text-xs opacity-80">Two players, one device</div>
              </div>
            </Button>
          </div>
        </div>

        {/* AI-specific settings */}
        {settings.mode === 'ai' && (
          <>
            {/* Difficulty Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">Difficulty</label>
              <div className="grid grid-cols-2 gap-2">
                {(Object.keys(DIFFICULTY_LEVELS) as DifficultyLevel[]).map((level) => (
                  <Button
                    key={level}
                    variant={settings.difficulty === level ? 'default' : 'outline'}
                    onClick={() => setSettings({ ...settings, difficulty: level })}
                    className="h-auto py-2"
                  >
                    <div className="text-center">
                      <div className="font-medium">{DIFFICULTY_LEVELS[level].name}</div>
                      <div className="text-xs opacity-80">{DIFFICULTY_LEVELS[level].description}</div>
                    </div>
                  </Button>
                ))}
              </div>
            </div>

            {/* Player Color Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">Play as</label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={settings.playerColor === 1 ? 'default' : 'outline'}
                  onClick={() => setSettings({ ...settings, playerColor: 1 })}
                  className="h-auto py-3"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-red-500" />
                    <div>
                      <div className="font-medium">Red</div>
                      <div className="text-xs opacity-80">Goes first</div>
                    </div>
                  </div>
                </Button>
                <Button
                  variant={settings.playerColor === 2 ? 'default' : 'outline'}
                  onClick={() => setSettings({ ...settings, playerColor: 2 })}
                  className="h-auto py-3"
                >
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded-full bg-yellow-400" />
                    <div>
                      <div className="font-medium">Yellow</div>
                      <div className="text-xs opacity-80">Goes second</div>
                    </div>
                  </div>
                </Button>
              </div>
            </div>
          </>
        )}

        {/* Start Button */}
        <Button onClick={handleStartGame} size="lg" className="w-full">
          Start Game
        </Button>
      </CardContent>
    </Card>
  )

  const renderGameScreen = () => (
    <Card>
      <CardHeader className="text-center pb-2">
        <div className="flex justify-center items-center gap-4 mb-2">
          {settings.mode === 'ai' ? (
            <>
              <div className="flex items-center gap-2">
                <div
                  className={`w-4 h-4 rounded-full ${userPlayerNumber === 1 ? 'bg-red-500' : 'bg-yellow-400'}`}
                />
                <span className="text-sm font-medium">You</span>
              </div>
              <span className="text-muted-foreground">vs</span>
              <div className="flex items-center gap-2">
                <div
                  className={`w-4 h-4 rounded-full ${aiPlayerNumber === 1 ? 'bg-red-500' : 'bg-yellow-400'}`}
                />
                <span className="text-sm font-medium">
                  AI ({DIFFICULTY_LEVELS[settings.difficulty].name})
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-red-500" />
                <span className="text-sm font-medium">Player 1</span>
              </div>
              <span className="text-muted-foreground">vs</span>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-yellow-400" />
                <span className="text-sm font-medium">Player 2</span>
              </div>
            </>
          )}
        </div>
        <CardTitle className={`text-2xl ${getStatusColor()}`}>{getStatusMessage()}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-6">
        <GameBoard
          board={gameState.board}
          currentPlayer={gameState.currentPlayer}
          winner={gameState.winner}
          onColumnClick={handleColumnClick}
          disabled={
            gameState.winner !== null ||
            (settings.mode === 'ai' && (gameState.currentPlayer !== userPlayerNumber || isBotThinking))
          }
          threats={threats}
          showThreats={showAnalysis}
        />

        {/* Analysis toggle and panel */}
        <div className="w-full space-y-3">
          <label className="flex items-center gap-2 cursor-pointer justify-center">
            <input
              type="checkbox"
              checked={showAnalysis}
              onChange={(e) => setShowAnalysis(e.target.checked)}
              className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-muted-foreground">Show analysis</span>
          </label>

          {showAnalysis && (
            <AnalysisPanel
              board={gameState.board}
              currentPlayer={gameState.currentPlayer}
              isGameOver={isGameOver}
            />
          )}
        </div>

        <div className="flex flex-col gap-3 w-full">
          {/* Game over actions */}
          {gameState.winner !== null && (
            <div className="flex flex-col gap-2">
              <div className="flex gap-2 justify-center">
                <Button onClick={handlePlayAgain} size="lg">
                  Play Again
                </Button>
                <Button onClick={handleNewGame} variant="outline" size="lg">
                  Change Settings
                </Button>
              </div>
              {isAuthenticated && !gameSaved && (
                <div className="flex justify-center">
                  <Button onClick={handleSaveGame} variant="outline" disabled={isSaving}>
                    {isSaving ? 'Saving...' : 'Save Game'}
                  </Button>
                </div>
              )}
              {isAuthenticated && gameSaved && (
                <p className="text-center text-sm text-green-600 dark:text-green-400">
                  Game saved to history!
                </p>
              )}
              {isAuthenticated && saveError && (
                <p className="text-center text-sm text-red-600 dark:text-red-400">{saveError}</p>
              )}
              {!isAuthenticated && (
                <p className="text-center text-sm text-muted-foreground">
                  <Link to="/login" className="text-primary hover:underline">
                    Sign in
                  </Link>{' '}
                  to save games and track your progress
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
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <Link to="/" className="text-2xl font-bold hover:opacity-80">
              MakeFour
            </Link>
            {user && <p className="text-xs text-muted-foreground">{user.email}</p>}
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
          {gamePhase === 'setup' ? renderSetupScreen() : renderGameScreen()}
        </div>
      </main>
    </div>
  )
}
