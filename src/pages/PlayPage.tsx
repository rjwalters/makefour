import { useState, useCallback, useEffect, useMemo, useRef } from 'react'
import { Link, useSearchParams, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { usePreferencesContext } from '../contexts/PreferencesContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import Navbar from '../components/Navbar'
import GameBoard from '../components/GameBoard'
import AnalysisPanel from '../components/AnalysisPanel'
import ChatPanel from '../components/ChatPanel'
import BotAvatar from '../components/BotAvatar'
import { GameTimers } from '../components/GameTimer'
import AutomatchWaiting from '../components/AutomatchWaiting'
import ChallengePlayer from '../components/ChallengePlayer'
import ChallengeWaiting from '../components/ChallengeWaiting'
import {
  createGameState,
  makeMove,
  getStateAtMove,
  replayMoves,
  type GameState,
  type Player,
} from '../game/makefour'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { useMatchmaking, type MatchmakingMode } from '../hooks/useMatchmaking'
import { useBotGame, type BotDifficulty } from '../hooks/useBotGame'
import { useChallenge } from '../hooks/useChallenge'
import { useSounds } from '../hooks/useSounds'
import type { BotPersona } from '../hooks/useBotPersonas'
import { useSingleBotStats } from '../hooks/usePlayerBotStats'
import { suggestMove, analyzeThreats, analyzePosition, DIFFICULTY_LEVELS, type DifficultyLevel, type Analysis } from '../ai/coach'

type GameMode = 'ai' | 'hotseat' | 'online'
type GamePhase = 'setup' | 'playing' | 'matchmaking' | 'online' | 'botGame' | 'competition-setup' | 'automatch' | 'challenging'
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
  const { isAuthenticated, user } = useAuth()
  const { preferences, updatePreferences } = usePreferencesContext()
  const { apiCall } = useAuthenticatedApi()
  const matchmaking = useMatchmaking()
  const botGame = useBotGame()
  const challenge = useChallenge()
  const sounds = useSounds()
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [gameState, setGameState] = useState<GameState>(createGameState)
  const [isSaving, setIsSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [gameSaved, setGameSaved] = useState(false)
  const [isBotThinking, setIsBotThinking] = useState(false)
  const [gamePhase, setGamePhase] = useState<GamePhase>('setup')
  const [showAnalysis, setShowAnalysis] = useState(false)
  const [hint, setHint] = useState<Analysis | null>(null)
  const [isGettingHint, setIsGettingHint] = useState(false)
  const [viewMoveIndex, setViewMoveIndex] = useState<number | null>(null) // null = live, number = viewing historical state
  const [selectedPersona, setSelectedPersona] = useState<BotPersona | null>(null)

  // Get the current bot persona ID (from active game or selected persona)
  const currentBotPersonaId = botGame.game?.botPersonaId ?? selectedPersona?.id ?? null
  const { data: currentBotStats } = useSingleBotStats(currentBotPersonaId)

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

  // Redirect to training mode if no mode specified
  useEffect(() => {
    const mode = searchParams.get('mode')
    if (!mode) {
      navigate('/play?mode=training', { replace: true })
    }
  }, [searchParams, navigate])

  // Handle mode query parameter from navbar navigation
  useEffect(() => {
    const mode = searchParams.get('mode')
    if (mode === 'training') {
      // Coach mode: AI training with hints/analysis
      // Cancel any active matchmaking/challenge when switching to training
      if (gamePhase === 'automatch' || gamePhase === 'matchmaking') {
        matchmaking.leaveQueue()
      }
      if (gamePhase === 'challenging') {
        challenge.cancelChallenge()
      }
      setSettings((prev) => ({
        ...prev,
        mode: 'ai',
        botGameMode: 'training',
      }))
      setGamePhase('setup')
    } else if (mode === 'compete') {
      // Compete mode: Show competition setup (Automatch vs Challenge)
      if (isAuthenticated) {
        // Only switch to competition-setup if not already in a competition phase
        if (gamePhase === 'setup') {
          setGamePhase('competition-setup')
        }
      } else {
        // Not authenticated - redirect to training mode instead
        navigate('/play?mode=training', { replace: true })
      }
    }
  }, [searchParams, gamePhase, isAuthenticated, navigate, matchmaking, challenge])

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

  // Handle challenge state changes
  useEffect(() => {
    if (challenge.status === 'matched' && challenge.matchedGame) {
      // Challenge was accepted - transition to online game
      // The matchmaking hook will handle game state polling
      // We need to simulate a matched state for the online game screen
      sounds.playMatchFound()
    }
  }, [challenge.status, challenge.matchedGame, sounds])

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

  // Compute the displayed game state (live or rewound)
  const displayedState = useMemo(() => {
    if (viewMoveIndex === null) {
      return gameState
    }
    return getStateAtMove(gameState.moveHistory, viewMoveIndex) ?? gameState
  }, [gameState, viewMoveIndex])

  // Whether we're viewing a historical state (rewound)
  const isViewingHistory = viewMoveIndex !== null && viewMoveIndex < gameState.moveHistory.length

  // Compute threats for highlighting (memoized for performance)
  const threats = useMemo(() => {
    if (!showAnalysis || isGameOver || isViewingHistory) return []
    const analysis = analyzeThreats(displayedState.board, displayedState.currentPlayer)
    return analysis.threats
  }, [displayedState.board, displayedState.currentPlayer, showAnalysis, isGameOver, isViewingHistory])

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
    if (isViewingHistory) return // Don't make AI moves while reviewing history
    if (gameState.currentPlayer !== aiPlayerNumber || gameState.winner !== null) return

    let cancelled = false
    setIsBotThinking(true)

    const makeBotMove = async () => {
      // Small delay for UX - makes it feel like the bot is "thinking"
      await new Promise((resolve) => setTimeout(resolve, 500))

      if (cancelled) return

      const botMove = await suggestMove(
        {
          board: gameState.board,
          currentPlayer: gameState.currentPlayer,
          moveHistory: gameState.moveHistory,
        },
        settings.difficulty
      )

      if (cancelled) return

      const newState = makeMove(gameState, botMove)
      if (newState) {
        setGameState(newState)
        sounds.playPieceDrop()
      }
      setIsBotThinking(false)
    }

    makeBotMove()

    return () => {
      cancelled = true
    }
  }, [gameState, gamePhase, settings.mode, settings.difficulty, aiPlayerNumber, sounds, isViewingHistory])

  const handleColumnClick = useCallback(
    (column: number) => {
      if (gamePhase !== 'playing') return

      // If viewing history, branch from that point
      if (isViewingHistory && viewMoveIndex !== null) {
        // Replay moves up to viewMoveIndex, then apply the new move
        const baseState = replayMoves(gameState.moveHistory.slice(0, viewMoveIndex))
        if (!baseState) return

        // In AI mode, only allow clicks on user's turn
        if (settings.mode === 'ai' && baseState.currentPlayer !== userPlayerNumber) {
          return
        }

        const newState = makeMove(baseState, column)
        if (newState) {
          setGameState(newState)
          setViewMoveIndex(null) // Return to live view
          setGameSaved(false)
          setSaveError(null)
          sounds.playPieceDrop()
        } else {
          sounds.playInvalidMove()
        }
        return
      }

      // Normal live play
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
    [gameState, gamePhase, settings.mode, userPlayerNumber, isBotThinking, sounds, isViewingHistory, viewMoveIndex]
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
      if (selectedPersona) {
        botGame.createGameWithPersona(selectedPersona.id, settings.playerColor)
      } else {
        botGame.createGame(settings.difficulty as BotDifficulty, settings.playerColor)
      }
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
    setHint(null)
    setViewMoveIndex(null)
    setSelectedPersona(null)
    matchmaking.reset()
    botGame.reset()
  }, [matchmaking, botGame])

  const handlePlayAgain = useCallback(() => {
    setGameState(createGameState())
    setGameSaved(false)
    setSaveError(null)
    setIsBotThinking(false)
    setHint(null)
    setViewMoveIndex(null)
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

  // Get hint (AI coach suggestion)
  const handleGetHint = useCallback(async () => {
    if (isGettingHint || gameState.winner !== null) return

    setIsGettingHint(true)
    try {
      const analysis = await analyzePosition(
        {
          board: gameState.board,
          currentPlayer: gameState.currentPlayer,
          moveHistory: gameState.moveHistory,
        },
        settings.difficulty
      )
      setHint(analysis)
    } catch (error) {
      console.error('Failed to get hint:', error)
    } finally {
      setIsGettingHint(false)
    }
  }, [gameState, settings.difficulty, isGettingHint])

  // Clear hint when game state changes (new move made)
  useEffect(() => {
    setHint(null)
  }, [gameState.moveHistory.length])

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
        <CardTitle className="text-2xl">Training Mode</CardTitle>
        <p className="text-sm text-muted-foreground mt-1">
          Practice with AI coaching, hints, and analysis
        </p>
      </CardHeader>
      <CardContent className="space-y-6">
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

        {/* Start Button */}
        <Button
          onClick={handleStartGame}
          size="lg"
          className="w-full"
        >
          Start Training
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
          board={displayedState.board}
          currentPlayer={displayedState.currentPlayer}
          winner={displayedState.winner}
          onColumnClick={handleColumnClick}
          disabled={
            (gameState.winner !== null && !isViewingHistory) ||
            (settings.mode === 'ai' && !isViewingHistory && (gameState.currentPlayer !== userPlayerNumber || isBotThinking))
          }
          threats={threats}
          showThreats={showAnalysis && !isViewingHistory}
        />

        {/* Rewind controls - only show when there are moves to review */}
        {gameState.moveHistory.length > 0 && (
          <div className="w-full flex flex-col items-center gap-2">
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setViewMoveIndex(0)}
                disabled={viewMoveIndex === 0}
                className="px-2"
                title="Go to start"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const current = viewMoveIndex ?? gameState.moveHistory.length
                  setViewMoveIndex(Math.max(0, current - 1))
                }}
                disabled={viewMoveIndex === 0}
                className="px-2"
                title="Previous move"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </Button>
              <span className="text-sm text-muted-foreground min-w-[80px] text-center">
                Move {viewMoveIndex ?? gameState.moveHistory.length} / {gameState.moveHistory.length}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  const current = viewMoveIndex ?? gameState.moveHistory.length
                  if (current >= gameState.moveHistory.length - 1) {
                    setViewMoveIndex(null) // Return to live
                  } else {
                    setViewMoveIndex(current + 1)
                  }
                }}
                disabled={viewMoveIndex === null}
                className="px-2"
                title="Next move"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setViewMoveIndex(null)}
                disabled={viewMoveIndex === null}
                className="px-2"
                title="Go to live"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
              </Button>
            </div>
            {isViewingHistory && (
              <p className="text-xs text-muted-foreground">
                Click a column to play from this position
              </p>
            )}
          </div>
        )}

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

          {showAnalysis && !isViewingHistory && (
            <AnalysisPanel
              board={displayedState.board}
              currentPlayer={displayedState.currentPlayer}
              isGameOver={isGameOver}
            />
          )}
        </div>

        {/* Get Hint button and display - only in AI training mode during player's turn (not when viewing history) */}
        {settings.mode === 'ai' && !isGameOver && !isViewingHistory && gameState.currentPlayer === userPlayerNumber && !isBotThinking && (
          <div className="w-full space-y-3">
            <Button
              onClick={handleGetHint}
              variant="secondary"
              disabled={isGettingHint}
              className="w-full"
            >
              {isGettingHint ? 'Analyzing...' : 'Get Hint'}
            </Button>

            {hint && (
              <div className="p-3 bg-muted rounded-lg text-sm">
                <p className="font-medium mb-1">AI Coach (experimental)</p>
                <p className="text-muted-foreground">{hint.evaluation}</p>
                {hint.bestMove >= 0 && (
                  <p className="mt-1 font-medium">
                    Suggested move: Column {hint.bestMove + 1}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

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

    // Use persona name if available, otherwise fall back to difficulty name
    const botName = game.botPersonaId
      ? selectedPersona?.name || 'Bot'
      : game.botDifficulty
        ? game.botDifficulty.charAt(0).toUpperCase() + game.botDifficulty.slice(1) + ' Bot'
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
              {selectedPersona?.avatarUrl ? (
                <BotAvatar
                  avatarUrl={selectedPersona.avatarUrl}
                  name={botName}
                  size="xs"
                />
              ) : (
                <div
                  className={`w-4 h-4 rounded-full ${game.playerNumber === 1 ? 'bg-yellow-400' : 'bg-red-500'}`}
                />
              )}
              <span className="text-sm font-medium">
                {botName} ({botColor}) - {game.opponentRating}
              </span>
            </div>
          </div>
          <CardTitle className={`text-2xl ${getBotStatusColor()}`}>
            {getBotStatusMessage()}
          </CardTitle>
          <p className="text-xs text-muted-foreground mt-1">
            Ranked Bot Match - 5 min clock
          </p>

          {/* Post-game stats display */}
          {isGameOver && currentBotStats && currentBotStats.hasPlayed && (
            <div className="mt-4 p-3 bg-muted/50 rounded-lg space-y-2">
              <div className="flex items-center justify-center gap-2">
                <span className="text-sm font-medium">Your record vs {botName}:</span>
                <span className="text-lg font-bold">
                  <span className="text-green-600 dark:text-green-400">{currentBotStats.wins}</span>
                  <span className="text-muted-foreground">-</span>
                  <span className="text-red-600 dark:text-red-400">{currentBotStats.losses}</span>
                  {currentBotStats.draws > 0 && (
                    <>
                      <span className="text-muted-foreground">-</span>
                      <span className="text-yellow-600 dark:text-yellow-400">{currentBotStats.draws}</span>
                    </>
                  )}
                </span>
              </div>

              {/* Streak notification */}
              {currentBotStats.currentStreak !== 0 && (
                <div className="text-center">
                  <span className={`text-sm px-2 py-1 rounded ${
                    currentBotStats.currentStreak > 0
                      ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                      : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                  }`}>
                    {currentBotStats.currentStreak > 0
                      ? `${currentBotStats.currentStreak}-game win streak against ${botName}!`
                      : `${Math.abs(currentBotStats.currentStreak)}-game loss streak`}
                  </span>
                </div>
              )}

              {/* First win celebration */}
              {game.winner === String(game.playerNumber) && currentBotStats.wins === 1 && currentBotStats.firstWinAt && (
                <div className="text-center">
                  <span className="text-sm px-2 py-1 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                    First win against {botName}!
                  </span>
                </div>
              )}

              {/* Mastery notification */}
              {currentBotStats.isMastered && (
                <div className="text-center">
                  <span className="text-sm px-2 py-1 rounded bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300">
                    You&apos;ve mastered {botName}!
                  </span>
                </div>
              )}
            </div>
          )}
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

          {/* Chat panel for ranked bot games */}
          <ChatPanel
            gameId={game.id}
            isActive={game.status === 'active' || game.status === 'completed'}
            isBot={true}
            playerNumber={game.playerNumber}
            moveCount={game.moves.length}
            className="w-full mt-4"
            botAvatarUrl={selectedPersona?.avatarUrl}
            botName={selectedPersona?.name || botName}
            userId={user?.id}
          />
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
            userId={user?.id}
          />
        </CardContent>
      </Card>
    )
  }

  const renderCompetitionSetupScreen = () => (
    <Card>
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">Competitive Play</CardTitle>
        <p className="text-muted-foreground mt-1">
          All matches affect your rating
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Automatch option */}
        <button
          onClick={() => {
            matchmaking.joinQueue('ranked', true)
            setGamePhase('automatch')
          }}
          className="w-full p-4 rounded-lg border-2 border-primary/20 hover:border-primary/50 hover:bg-primary/5 transition-all text-left"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <div>
              <h3 className="font-bold text-lg">Automatch</h3>
              <p className="text-sm text-muted-foreground">
                Find an opponent matched to your rating. Plays a human if available, or a bot at your level.
              </p>
            </div>
          </div>
        </button>

        {/* Challenge option */}
        <button
          onClick={() => setGamePhase('challenging')}
          className="w-full p-4 rounded-lg border-2 border-primary/20 hover:border-primary/50 hover:bg-primary/5 transition-all text-left"
        >
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
              <svg className="w-6 h-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <div>
              <h3 className="font-bold text-lg">Challenge Player</h3>
              <p className="text-sm text-muted-foreground">
                Challenge a specific player by username. Both players must accept for the match to start.
              </p>
            </div>
          </div>
        </button>

        {/* Back button */}
        <Button
          variant="outline"
          onClick={() => setGamePhase('setup')}
          className="w-full mt-4"
        >
          Back to Game Options
        </Button>
      </CardContent>
    </Card>
  )

  const renderAutomatchScreen = () => (
    <AutomatchWaiting
      waitTime={Math.floor(matchmaking.waitTime / 1000)}
      currentTolerance={matchmaking.currentTolerance}
      userRating={matchmaking.userRating}
      onCancel={() => {
        matchmaking.leaveQueue()
        setGamePhase('competition-setup')
      }}
      onPlayBotNow={() => {
        matchmaking.playBotNow()
      }}
    />
  )

  const renderChallengingScreen = () => {
    if (challenge.status === 'waiting' && challenge.outgoingChallenge) {
      return (
        <ChallengeWaiting
          targetUsername={challenge.outgoingChallenge.targetUsername}
          targetRating={challenge.outgoingChallenge.targetRating}
          targetExists={challenge.outgoingChallenge.targetExists}
          expiresAt={challenge.outgoingChallenge.expiresAt}
          onCancel={() => {
            challenge.cancelChallenge()
            setGamePhase('challenging')
          }}
        />
      )
    }

    return (
      <ChallengePlayer
        onSendChallenge={(username) => challenge.sendChallenge(username)}
        onBack={() => setGamePhase('competition-setup')}
        isLoading={challenge.status === 'sending'}
        error={challenge.error}
      />
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
      case 'competition-setup':
        return renderCompetitionSetupScreen()
      case 'automatch':
        return renderAutomatchScreen()
      case 'challenging':
        return renderChallengingScreen()
      default:
        return renderSetupScreen()
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-lg mx-auto">
          {renderContent()}
        </div>
      </main>
    </div>
  )
}
