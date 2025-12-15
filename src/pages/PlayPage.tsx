import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { usePreferencesContext } from '../contexts/PreferencesContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import SoundToggle from '../components/SoundToggle'
import GameBoard from '../components/GameBoard'
import AnalysisPanel from '../components/AnalysisPanel'
import ChatPanel from '../components/ChatPanel'
import { GameTimers } from '../components/GameTimer'
import {
  createGameState,
  makeMove,
  type GameState,
  type Player,
} from '../game/makefour'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { useMatchmaking, type MatchmakingMode } from '../hooks/useMatchmaking'
import { useBotGame, type BotDifficulty } from '../hooks/useBotGame'
import { useSounds } from '../hooks/useSounds'
import { suggestMove, analyzeThreats, DIFFICULTY_LEVELS, type DifficultyLevel } from '../ai/coach'

type GameMode = 'ai' | 'hotseat' | 'online'
type GamePhase = 'setup' | 'playing' | 'matchmaking' | 'online' | 'botGame'
type BotGameMode = 'training' | 'ranked'

interface GameSettings {
  mode: GameMode
  difficulty: DifficultyLevel
  playerColor: Player // Which player the user plays as (1 = red/first, 2 = yellow/second)
  matchmakingMode: MatchmakingMode
  allowSpectators: boolean // Whether spectators can watch your online games
  botGameMode: BotGameMode // Training (untimed, no ELO) or Ranked (timed, affects ELO)
}

export default function PlayPage() {
  const { logout, user, isAuthenticated } = useAuth()
  const { preferences, updatePreferences } = usePreferencesContext()
  const { apiCall } = useAuthenticatedApi()
  const matchmaking = useMatchmaking()
  const botGame = useBotGame()
  const sounds = useSounds()
  const [gameState, setGameState] = useState<GameState>(createGameState)
  const [isSaving, setIsSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [gameSaved, setGameSaved] = useState(false)
  const [isBotThinking, setIsBotThinking] = useState(false)
  const [gamePhase, setGamePhase] = useState<GamePhase>('setup')
  const [showAnalysis, setShowAnalysis] = useState(false)

  // Derive settings from preferences (local state for in-session changes before game starts)
  const [settings, setSettings] = useState<GameSettings>(() => ({
    mode: preferences.defaultGameMode,
    difficulty: preferences.defaultDifficulty,
    playerColor: preferences.defaultPlayerColor,
    matchmakingMode: preferences.defaultMatchmakingMode,
    allowSpectators: preferences.allowSpectators,
    botGameMode: 'training',
  }))

  // Sync settings with preferences when preferences load/change (only in setup phase)
  useEffect(() => {
    if (gamePhase === 'setup') {
      setSettings((prev) => ({
        ...prev,
        mode: preferences.defaultGameMode,
        difficulty: preferences.defaultDifficulty,
        playerColor: preferences.defaultPlayerColor,
        matchmakingMode: preferences.defaultMatchmakingMode,
        allowSpectators: preferences.allowSpectators,
      }))
    }
  }, [preferences, gamePhase])

  // Track previous game state for detecting game over transitions
  const prevWinnerRef = useRef<typeof gameState.winner>(null)

  // Determine player numbers based on settings
  const userPlayerNumber = settings.playerColor
  const aiPlayerNumber: Player = settings.playerColor === 1 ? 2 : 1
  const isGameOver = gameState.winner !== null

  // Track previous matchmaking status for detecting match found
  const prevMatchmakingStatusRef = useRef(matchmaking.status)

  // Handle matchmaking state changes
  useEffect(() => {
    const prevStatus = prevMatchmakingStatusRef.current
    prevMatchmakingStatusRef.current = matchmaking.status

    if (matchmaking.status === 'playing' || matchmaking.status === 'completed') {
      setGamePhase('online')
      // Play match found sound when transitioning from queued/matched to playing
      if ((prevStatus === 'queued' || prevStatus === 'matched') && matchmaking.status === 'playing') {
        sounds.playMatchFound()
      }
    }
  }, [matchmaking.status, sounds])

  // Handle bot game state changes
  useEffect(() => {
    if (botGame.status === 'playing' || botGame.status === 'completed') {
      setGamePhase('botGame')
    }
  }, [botGame.status])

  // Track online game state for sounds
  const prevOnlineGameRef = useRef<{ winner: string | null; moveCount: number }>({
    winner: null,
    moveCount: 0,
  })

  // Play sounds for online game events
  useEffect(() => {
    if (!matchmaking.game) return

    const prevState = prevOnlineGameRef.current
    const currentMoveCount = matchmaking.game.moves.length

    // Play piece drop when opponent makes a move
    if (currentMoveCount > prevState.moveCount && !matchmaking.game.isYourTurn) {
      sounds.playPieceDrop()
    }

    // Play game over sounds
    if (matchmaking.game.winner && matchmaking.game.winner !== prevState.winner) {
      if (matchmaking.game.winner === 'draw') {
        sounds.playDraw()
      } else if (matchmaking.game.winner === String(matchmaking.game.playerNumber)) {
        sounds.playWin()
      } else {
        sounds.playLose()
      }
    }

    prevOnlineGameRef.current = {
      winner: matchmaking.game.winner,
      moveCount: currentMoveCount,
    }
  }, [matchmaking.game, sounds])

  // Compute threats for highlighting (memoized for performance)
  const threats = useMemo(() => {
    if (!showAnalysis || isGameOver) return []
    const analysis = analyzeThreats(gameState.board, gameState.currentPlayer)
    return analysis.threats
  }, [gameState.board, gameState.currentPlayer, showAnalysis, isGameOver])

  // Play game over sounds when winner changes
  useEffect(() => {
    if (prevWinnerRef.current === gameState.winner) return
    prevWinnerRef.current = gameState.winner

    if (gameState.winner === null) return

    if (gameState.winner === 'draw') {
      sounds.playDraw()
    } else if (settings.mode === 'hotseat') {
      // In hotseat, just play a generic win sound
      sounds.playWin()
    } else if (gameState.winner === userPlayerNumber) {
      sounds.playWin()
    } else {
      sounds.playLose()
    }
  }, [gameState.winner, settings.mode, userPlayerNumber, sounds])

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
        sounds.playPieceDrop()
      }
      setIsBotThinking(false)
    }

    makeBotMove()
  }, [gameState, gamePhase, settings.mode, settings.difficulty, aiPlayerNumber, sounds])

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
        sounds.playPieceDrop()
      } else {
        sounds.playInvalidMove()
      }
    },
    [gameState, gamePhase, settings.mode, userPlayerNumber, isBotThinking, sounds]
  )

  const handleStartGame = useCallback(() => {
    // Save settings to centralized preferences
    updatePreferences({
      defaultGameMode: settings.mode,
      defaultDifficulty: settings.difficulty,
      defaultPlayerColor: settings.playerColor,
      defaultMatchmakingMode: settings.matchmakingMode,
      allowSpectators: settings.allowSpectators,
    })

    setGameState(createGameState())
    setGameSaved(false)
    setSaveError(null)
    setIsBotThinking(false)

    if (settings.mode === 'online') {
      // Start matchmaking with spectator preference
      setGamePhase('matchmaking')
      matchmaking.joinQueue(settings.matchmakingMode, settings.allowSpectators)
    } else if (settings.mode === 'ai' && settings.botGameMode === 'ranked') {
      // Start ranked bot game (server-side, timed)
      botGame.createGame(settings.difficulty as BotDifficulty, settings.playerColor)
      sounds.playGameStart()
    } else {
      // Training mode or hotseat (client-side)
      setGamePhase('playing')
      sounds.playGameStart()
    }
  }, [settings, matchmaking, botGame, sounds, updatePreferences])

  const handleNewGame = useCallback(() => {
    setGamePhase('setup')
    setGameState(createGameState())
    setGameSaved(false)
    setSaveError(null)
    setIsBotThinking(false)
    matchmaking.reset()
    botGame.reset()
  }, [matchmaking, botGame])

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

  // Handle online game column click
  const handleOnlineColumnClick = useCallback(
    async (column: number) => {
      if (!matchmaking.game) return
      if (matchmaking.game.status !== 'active') return
      if (!matchmaking.game.isYourTurn) return

      await matchmaking.submitMove(column)
      sounds.playPieceDrop()
    },
    [matchmaking, sounds]
  )

  // Handle bot game column click
  const handleBotGameColumnClick = useCallback(
    async (column: number) => {
      if (!botGame.game) return
      if (botGame.game.status !== 'active') return
      if (!botGame.game.isYourTurn) return

      await botGame.submitMove(column)
      sounds.playPieceDrop()
    },
    [botGame, sounds]
  )

  // Cancel matchmaking
  const handleCancelMatchmaking = useCallback(() => {
    matchmaking.leaveQueue()
    setGamePhase('setup')
  }, [matchmaking])

  // Resign from online game
  const handleResign = useCallback(() => {
    matchmaking.resign()
  }, [matchmaking])

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
          <div className="grid grid-cols-3 gap-2">
            <Button
              variant={settings.mode === 'ai' ? 'default' : 'outline'}
              onClick={() => setSettings({ ...settings, mode: 'ai' })}
              className="h-auto py-3"
            >
              <div className="text-center">
                <div className="font-medium">vs AI</div>
                <div className="text-xs opacity-80">Computer</div>
              </div>
            </Button>
            <Button
              variant={settings.mode === 'online' ? 'default' : 'outline'}
              onClick={() => setSettings({ ...settings, mode: 'online' })}
              className="h-auto py-3"
            >
              <div className="text-center">
                <div className="font-medium">Online</div>
                <div className="text-xs opacity-80">Real player</div>
              </div>
            </Button>
            <Button
              variant={settings.mode === 'hotseat' ? 'default' : 'outline'}
              onClick={() => setSettings({ ...settings, mode: 'hotseat' })}
              className="h-auto py-3"
            >
              <div className="text-center">
                <div className="font-medium">Hotseat</div>
                <div className="text-xs opacity-80">Same device</div>
              </div>
            </Button>
          </div>
        </div>

        {/* AI-specific settings */}
        {settings.mode === 'ai' && (
          <>
            {/* Game Type Selection: Training vs Ranked */}
            <div>
              <label className="block text-sm font-medium mb-2">Game Type</label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={settings.botGameMode === 'training' ? 'default' : 'outline'}
                  onClick={() => setSettings({ ...settings, botGameMode: 'training' })}
                  className="h-auto py-3"
                >
                  <div className="text-center">
                    <div className="font-medium">Training</div>
                    <div className="text-xs opacity-80">Untimed, no rating</div>
                  </div>
                </Button>
                <Button
                  variant={settings.botGameMode === 'ranked' ? 'default' : 'outline'}
                  onClick={() => setSettings({ ...settings, botGameMode: 'ranked' })}
                  className="h-auto py-3"
                  disabled={!isAuthenticated}
                >
                  <div className="text-center">
                    <div className="font-medium">Ranked</div>
                    <div className="text-xs opacity-80">5-min clock, affects ELO</div>
                  </div>
                </Button>
              </div>
              {settings.botGameMode === 'ranked' && (
                <p className="text-xs text-muted-foreground mt-2">
                  Ranked bot games use a 5-minute chess clock and affect your rating.
                </p>
              )}
              {settings.botGameMode === 'ranked' && !isAuthenticated && (
                <p className="text-xs text-orange-600 dark:text-orange-400 mt-2">
                  Sign in to play ranked bot games.
                </p>
              )}
            </div>

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

        {/* Online-specific settings */}
        {settings.mode === 'online' && (
          <>
            <div>
              <label className="block text-sm font-medium mb-2">Match Type</label>
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant={settings.matchmakingMode === 'ranked' ? 'default' : 'outline'}
                  onClick={() => setSettings({ ...settings, matchmakingMode: 'ranked' })}
                  className="h-auto py-3"
                >
                  <div className="text-center">
                    <div className="font-medium">Ranked</div>
                    <div className="text-xs opacity-80">Affects your rating</div>
                  </div>
                </Button>
                <Button
                  variant={settings.matchmakingMode === 'casual' ? 'default' : 'outline'}
                  onClick={() => setSettings({ ...settings, matchmakingMode: 'casual' })}
                  className="h-auto py-3"
                >
                  <div className="text-center">
                    <div className="font-medium">Casual</div>
                    <div className="text-xs opacity-80">Just for fun</div>
                  </div>
                </Button>
              </div>
            </div>

            {/* Privacy settings */}
            <div>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={settings.allowSpectators}
                  onChange={(e) => setSettings({ ...settings, allowSpectators: e.target.checked })}
                  className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <div>
                  <span className="text-sm font-medium">Allow spectators</span>
                  <p className="text-xs text-muted-foreground">
                    Let others watch your game in real-time
                  </p>
                </div>
              </label>
            </div>

            {/* Watch games link */}
            <Link to="/spectate" className="block">
              <Button variant="outline" className="w-full">
                Watch Live Games
              </Button>
            </Link>
          </>
        )}

        {/* Login prompt for online mode */}
        {settings.mode === 'online' && !isAuthenticated && (
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-4 text-center">
            <p className="text-sm text-yellow-800 dark:text-yellow-200 mb-2">
              Sign in to play online matches
            </p>
            <Link to="/login">
              <Button variant="outline" size="sm">
                Sign In
              </Button>
            </Link>
          </div>
        )}

        {/* Start Button */}
        <Button
          onClick={handleStartGame}
          size="lg"
          className="w-full"
          disabled={settings.mode === 'online' && !isAuthenticated}
        >
          {settings.mode === 'online' ? 'Find Match' : 'Start Game'}
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

  const renderMatchmakingScreen = () => {
    const waitSeconds = Math.floor(matchmaking.waitTime / 1000)
    const waitMinutes = Math.floor(waitSeconds / 60)
    const waitDisplay = waitMinutes > 0
      ? `${waitMinutes}:${String(waitSeconds % 60).padStart(2, '0')}`
      : `${waitSeconds}s`

    return (
      <Card>
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Finding Opponent</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Loading animation */}
          <div className="flex justify-center">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          </div>

          {/* Status info */}
          <div className="text-center space-y-2">
            <p className="text-lg font-medium">
              {settings.matchmakingMode === 'ranked' ? 'Ranked Match' : 'Casual Match'}
            </p>
            <p className="text-sm text-muted-foreground">
              Searching for players with similar rating...
            </p>
            <div className="flex justify-center gap-4 text-sm text-muted-foreground">
              <span>Wait time: {waitDisplay}</span>
              <span>Rating range: ±{matchmaking.currentTolerance}</span>
            </div>
          </div>

          {/* Error message */}
          {matchmaking.error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-center">
              <p className="text-sm text-red-800 dark:text-red-200">{matchmaking.error}</p>
            </div>
          )}

          {/* Cancel button */}
          <Button
            onClick={handleCancelMatchmaking}
            variant="outline"
            size="lg"
            className="w-full"
          >
            Cancel
          </Button>
        </CardContent>
      </Card>
    )
  }

  const renderBotGameScreen = () => {
    const game = botGame.game
    if (!game) return null

    const isGameOver = game.status === 'completed'
    const playerColor = game.playerNumber === 1 ? 'Red' : 'Yellow'
    const botColor = game.playerNumber === 1 ? 'Yellow' : 'Red'

    const getBotStatusMessage = (): string => {
      if (game.winner === 'draw') return "It's a draw!"
      if (game.winner === String(game.playerNumber)) {
        // Check if bot ran out of time
        const botTimeMs = game.playerNumber === 1 ? game.player2TimeMs : game.player1TimeMs
        if (botTimeMs !== null && botTimeMs <= 0) {
          return 'You win! (Bot timed out)'
        }
        return 'You win!'
      }
      if (game.winner) {
        // Check if we ran out of time
        const ourTimeMs = game.playerNumber === 1 ? game.player1TimeMs : game.player2TimeMs
        if (ourTimeMs !== null && ourTimeMs <= 0) {
          return 'Time ran out!'
        }
        return 'Bot wins!'
      }
      if (game.isYourTurn) return 'Your turn'
      return 'Bot is thinking...'
    }

    const getBotStatusColor = (): string => {
      if (game.winner === 'draw') return 'text-muted-foreground'
      if (game.winner === String(game.playerNumber)) return 'text-green-600 dark:text-green-400'
      if (game.winner) return 'text-red-500'
      if (game.isYourTurn) {
        return game.playerNumber === 1 ? 'text-red-500' : 'text-yellow-500'
      }
      return 'text-yellow-500'
    }

    const difficultyName = game.botDifficulty
      ? game.botDifficulty.charAt(0).toUpperCase() + game.botDifficulty.slice(1)
      : 'Bot'

    return (
      <Card>
        <CardHeader className="text-center pb-2">
          <div className="flex justify-center items-center gap-4 mb-2">
            <div className="flex items-center gap-2">
              <div
                className={`w-4 h-4 rounded-full ${game.playerNumber === 1 ? 'bg-red-500' : 'bg-yellow-400'}`}
              />
              <span className="text-sm font-medium">You ({playerColor})</span>
            </div>
            <span className="text-muted-foreground">vs</span>
            <div className="flex items-center gap-2">
              <div
                className={`w-4 h-4 rounded-full ${game.playerNumber === 1 ? 'bg-yellow-400' : 'bg-red-500'}`}
              />
              <span className="text-sm font-medium">
                {difficultyName} Bot ({botColor}) - {game.opponentRating}
              </span>
            </div>
          </div>
          <CardTitle className={`text-2xl ${getBotStatusColor()}`}>
            {getBotStatusMessage()}
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Ranked Bot Match - 5 min clock
          </p>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-6">
          {/* Game Timers */}
          {game.timeControlMs !== null && (
            <GameTimers
              player1TimeMs={game.player1TimeMs}
              player2TimeMs={game.player2TimeMs}
              turnStartedAt={game.turnStartedAt}
              currentTurn={game.currentTurn}
              playerNumber={game.playerNumber}
              gameStatus={game.status}
              className="w-full max-w-xs"
            />
          )}

          {game.board && (
            <GameBoard
              board={game.board}
              currentPlayer={game.currentTurn as 1 | 2}
              winner={
                game.winner === 'draw'
                  ? 'draw'
                  : game.winner === '1'
                    ? 1
                    : game.winner === '2'
                      ? 2
                      : null
              }
              onColumnClick={handleBotGameColumnClick}
              disabled={!game.isYourTurn || isGameOver}
              threats={[]}
              showThreats={false}
            />
          )}

          <div className="flex flex-col gap-3 w-full">
            {/* Game over actions */}
            {isGameOver && (
              <div className="flex flex-col gap-2">
                <div className="flex gap-2 justify-center">
                  <Button onClick={handleNewGame} size="lg">
                    New Game
                  </Button>
                </div>
              </div>
            )}

            {/* Mid-game actions */}
            {!isGameOver && (
              <div className="flex gap-2 justify-center">
                <Button onClick={handleNewGame} variant="outline" size="sm">
                  Abandon Game
                </Button>
              </div>
            )}

            {/* Move counter */}
            <p className="text-center text-sm text-muted-foreground">
              Moves: {game.moves.length}
            </p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const renderOnlineGameScreen = () => {
    const game = matchmaking.game
    if (!game) return null

    const isGameOver = game.status === 'completed'
    const playerColor = game.playerNumber === 1 ? 'Red' : 'Yellow'
    const opponentColor = game.playerNumber === 1 ? 'Yellow' : 'Red'

    const getOnlineStatusMessage = (): string => {
      if (game.winner === 'draw') return "It's a draw!"
      if (game.winner === String(game.playerNumber)) {
        // Check if opponent ran out of time
        const opponentTimeMs = game.playerNumber === 1 ? game.player2TimeMs : game.player1TimeMs
        if (opponentTimeMs !== null && opponentTimeMs <= 0) {
          return 'You win! (Opponent timed out)'
        }
        return 'You win!'
      }
      if (game.winner) {
        // Check if we ran out of time
        const ourTimeMs = game.playerNumber === 1 ? game.player1TimeMs : game.player2TimeMs
        if (ourTimeMs !== null && ourTimeMs <= 0) {
          return 'Time ran out!'
        }
        return 'Opponent wins!'
      }
      if (game.isYourTurn) return 'Your turn'
      return "Opponent's turn..."
    }

    const getOnlineStatusColor = (): string => {
      if (game.winner === 'draw') return 'text-muted-foreground'
      if (game.winner === String(game.playerNumber)) return 'text-green-600 dark:text-green-400'
      if (game.winner) return 'text-red-500'
      if (game.isYourTurn) {
        return game.playerNumber === 1 ? 'text-red-500' : 'text-yellow-500'
      }
      return 'text-muted-foreground'
    }

    return (
      <Card>
        <CardHeader className="text-center pb-2">
          <div className="flex justify-center items-center gap-4 mb-2">
            <div className="flex items-center gap-2">
              <div
                className={`w-4 h-4 rounded-full ${game.playerNumber === 1 ? 'bg-red-500' : 'bg-yellow-400'}`}
              />
              <span className="text-sm font-medium">You ({playerColor})</span>
            </div>
            <span className="text-muted-foreground">vs</span>
            <div className="flex items-center gap-2">
              <div
                className={`w-4 h-4 rounded-full ${game.playerNumber === 1 ? 'bg-yellow-400' : 'bg-red-500'}`}
              />
              <span className="text-sm font-medium">
                Opponent ({opponentColor}) - {game.opponentRating}
              </span>
            </div>
          </div>
          <CardTitle className={`text-2xl ${getOnlineStatusColor()}`}>
            {getOnlineStatusMessage()}
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            {game.mode === 'ranked' ? 'Ranked Match' : 'Casual Match'}
          </p>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-6">
          {/* Game Timers */}
          {game.timeControlMs !== null && (
            <GameTimers
              player1TimeMs={game.player1TimeMs}
              player2TimeMs={game.player2TimeMs}
              turnStartedAt={game.turnStartedAt}
              currentTurn={game.currentTurn}
              playerNumber={game.playerNumber}
              gameStatus={game.status}
              className="w-full max-w-xs"
            />
          )}

          {game.board && (
            <GameBoard
              board={game.board}
              currentPlayer={game.currentTurn as 1 | 2}
              winner={
                game.winner === 'draw'
                  ? 'draw'
                  : game.winner === '1'
                    ? 1
                    : game.winner === '2'
                      ? 2
                      : null
              }
              onColumnClick={handleOnlineColumnClick}
              disabled={!game.isYourTurn || isGameOver}
              threats={[]}
              showThreats={false}
            />
          )}

          <div className="flex flex-col gap-3 w-full">
            {/* Game over actions */}
            {isGameOver && (
              <div className="flex flex-col gap-2">
                <div className="flex gap-2 justify-center">
                  <Button onClick={handleNewGame} size="lg">
                    New Game
                  </Button>
                </div>
              </div>
            )}

            {/* Mid-game actions */}
            {!isGameOver && (
              <div className="flex gap-2 justify-center">
                <Button onClick={handleResign} variant="outline" size="sm">
                  Resign
                </Button>
              </div>
            )}

            {/* Move counter */}
            <p className="text-center text-sm text-muted-foreground">
              Moves: {game.moves.length}
            </p>
          </div>

          {/* Chat panel for online games */}
          <ChatPanel
            gameId={game.id}
            isActive={game.status === 'active'}
            isBot={false}
            playerNumber={game.playerNumber}
            className="w-full mt-4"
          />
        </CardContent>
      </Card>
    )
  }

  const renderContent = () => {
    switch (gamePhase) {
      case 'setup':
        return renderSetupScreen()
      case 'playing':
        return renderGameScreen()
      case 'matchmaking':
        return renderMatchmakingScreen()
      case 'online':
        return renderOnlineGameScreen()
      case 'botGame':
        return renderBotGameScreen()
      default:
        return renderSetupScreen()
    }
  }

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
                <SoundToggle
                  settings={sounds.settings}
                  onToggle={sounds.toggleSound}
                  onVolumeChange={(volume) => sounds.updateSettings({ volume })}
                />
                <ThemeToggle />
                <Button variant="outline" onClick={logout} size="sm">
                  Logout
                </Button>
              </>
            ) : (
              <>
                <SoundToggle
                  settings={sounds.settings}
                  onToggle={sounds.toggleSound}
                  onVolumeChange={(volume) => sounds.updateSettings({ volume })}
                />
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
          {renderContent()}
        </div>
      </main>
    </div>
  )
}
