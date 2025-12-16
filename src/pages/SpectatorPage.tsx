import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import Navbar from '../components/Navbar'
import GameBoard from '../components/GameBoard'
import { SpectatorTimers } from '../components/GameTimer'
import { useSpectate, type LiveGame } from '../hooks/useSpectate'
import { useSpectatorChat } from '../hooks/useSpectatorChat'
import { useBotBattlePlayback } from '../hooks/useBotBattlePlayback'

type ViewMode = 'live' | 'bot-battle'

export default function SpectatorPage() {
  const [viewMode, setViewMode] = useState<ViewMode>('live')
  const spectator = useSpectate()
  const botBattle = useBotBattlePlayback()
  const chat = useSpectatorChat(
    spectator.currentGame?.id || null,
    spectator.status === 'watching' && (spectator.currentGame?.isBotVsBot ?? false)
  )

  // Auto-start browsing when page loads (only for live mode)
  useEffect(() => {
    if (viewMode === 'live' && spectator.status === 'idle') {
      spectator.startBrowsing()
    }
  }, [viewMode, spectator.status, spectator.startBrowsing])

  // Handle switching to bot battle mode
  const handleStartBotBattle = async () => {
    setViewMode('bot-battle')
    spectator.stopBrowsing()
    await botBattle.generateGame()
  }

  // Handle going back to live games
  const handleBackToLive = () => {
    botBattle.reset()
    setViewMode('live')
    spectator.startBrowsing()
  }

  const renderGameCard = (game: LiveGame) => {
    const avgRating = Math.round((game.player1.rating + game.player2.rating) / 2)
    const duration = Math.floor((Date.now() - game.createdAt) / 60000)
    const isBotVsBot = game.isBotVsBot ?? false

    return (
      <Card
        key={game.id}
        className={`cursor-pointer hover:border-primary transition-colors ${
          isBotVsBot ? 'border-purple-500/50 bg-purple-50/30 dark:bg-purple-950/20' : ''
        }`}
        onClick={() => spectator.watchGame(game.id)}
      >
        <CardContent className="p-4">
          {/* Bot vs Bot Badge */}
          {isBotVsBot && (
            <div className="flex items-center gap-1 mb-2">
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.616 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.715-5.349L11 6.477V16h2a1 1 0 110 2H7a1 1 0 110-2h2V6.477L6.237 7.582l1.715 5.349a1 1 0 01-.285 1.05A3.989 3.989 0 015 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.788l1.599.799L9 4.323V3a1 1 0 011-1zm-5 8.274l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L5 10.274zm10 0l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L15 10.274z" />
                </svg>
                Bot Battle
              </span>
            </div>
          )}
          <div className="flex justify-between items-center mb-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-sm font-medium">{game.player1.displayName}</span>
              <span className="text-xs text-muted-foreground">({game.player1.rating})</span>
            </div>
            <span className="text-xs text-muted-foreground">vs</span>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">({game.player2.rating})</span>
              <span className="text-sm font-medium">{game.player2.displayName}</span>
              <div className="w-3 h-3 rounded-full bg-yellow-400" />
            </div>
          </div>
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              {game.mode === 'ranked' ? 'Ranked' : 'Casual'} - Avg {avgRating}
            </span>
            <span>{game.moveCount} moves</span>
            <span>{duration}m</span>
            <span>{game.spectatorCount} watching</span>
          </div>
          <div className="mt-2 flex items-center gap-1">
            <div
              className={`w-2 h-2 rounded-full ${
                game.currentTurn === 1 ? 'bg-red-500' : 'bg-yellow-400'
              }`}
            />
            <span className="text-xs">
              {game.currentTurn === 1 ? game.player1.displayName : game.player2.displayName}'s turn
            </span>
          </div>
        </CardContent>
      </Card>
    )
  }

  const renderBrowsingScreen = () => (
    <Card>
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">Live Games</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {spectator.isLoading && (
          <div className="flex justify-center py-8">
            <div className="w-8 h-8 border-4 border-primary border-t-transparent rounded-full animate-spin" />
          </div>
        )}

        {!spectator.isLoading && spectator.liveGames.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            <p>No live games available right now.</p>
            <p className="text-sm mt-2">Check back later or start your own game!</p>
          </div>
        )}

        {!spectator.isLoading && spectator.liveGames.length > 0 && (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground text-center">
              {spectator.totalGames} game{spectator.totalGames !== 1 ? 's' : ''} in progress
            </p>
            {spectator.liveGames.map(renderGameCard)}
          </div>
        )}

        {spectator.error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 text-center">
            <p className="text-sm text-red-800 dark:text-red-200">{spectator.error}</p>
          </div>
        )}

        {/* Bot Battle CTA */}
        <div className="border-t pt-4 mt-4">
          <div className="text-center mb-3">
            <p className="text-sm text-muted-foreground">
              Or watch AI bots battle each other
            </p>
          </div>
          <Button
            onClick={handleStartBotBattle}
            className="w-full bg-purple-600 hover:bg-purple-700"
          >
            <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.616 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.715-5.349L11 6.477V16h2a1 1 0 110 2H7a1 1 0 110-2h2V6.477L6.237 7.582l1.715 5.349a1 1 0 01-.285 1.05A3.989 3.989 0 015 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.788l1.599.799L9 4.323V3a1 1 0 011-1zm-5 8.274l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L5 10.274zm10 0l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L15 10.274z" />
            </svg>
            Watch Bot Battle
          </Button>
        </div>

        <Link to="/play" className="block">
          <Button variant="outline" className="w-full">
            Back to Play
          </Button>
        </Link>
      </CardContent>
    </Card>
  )

  const renderWatchingScreen = () => {
    const game = spectator.currentGame
    if (!game) return null

    const isGameOver = game.status !== 'active'
    const isBotVsBot = game.isBotVsBot ?? false

    const getStatusMessage = (): string => {
      if (game.winner === 'draw') return "It's a draw!"
      if (game.winner === '1') return `${game.player1.displayName} wins!`
      if (game.winner === '2') return `${game.player2.displayName} wins!`
      if (game.currentTurn === 1) return `${game.player1.displayName}'s turn`
      return `${game.player2.displayName}'s turn`
    }

    const getStatusColor = (): string => {
      if (game.winner === 'draw') return 'text-muted-foreground'
      if (game.winner) return 'text-green-600 dark:text-green-400'
      return game.currentTurn === 1 ? 'text-red-500' : 'text-yellow-500'
    }

    return (
      <Card className={isBotVsBot ? 'border-purple-500/50' : ''}>
        <CardHeader className="text-center pb-2">
          {/* Bot vs Bot Badge */}
          {isBotVsBot && (
            <div className="flex justify-center mb-2">
              <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.616 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.715-5.349L11 6.477V16h2a1 1 0 110 2H7a1 1 0 110-2h2V6.477L6.237 7.582l1.715 5.349a1 1 0 01-.285 1.05A3.989 3.989 0 015 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.788l1.599.799L9 4.323V3a1 1 0 011-1zm-5 8.274l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L5 10.274zm10 0l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L15 10.274z" />
                </svg>
                Bot Battle
              </span>
            </div>
          )}
          <div className="flex justify-center items-center gap-4 mb-2">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500" />
              <div className="text-left">
                <span className="text-sm font-medium">{game.player1.displayName}</span>
                <span className="text-xs text-muted-foreground ml-1">({game.player1.rating})</span>
              </div>
            </div>
            <span className="text-muted-foreground">vs</span>
            <div className="flex items-center gap-2">
              <div className="text-right">
                <span className="text-sm font-medium">{game.player2.displayName}</span>
                <span className="text-xs text-muted-foreground ml-1">({game.player2.rating})</span>
              </div>
              <div className="w-4 h-4 rounded-full bg-yellow-400" />
            </div>
          </div>
          <CardTitle className={`text-2xl ${getStatusColor()}`}>{getStatusMessage()}</CardTitle>
          <div className="flex justify-center gap-4 text-xs text-muted-foreground mt-1">
            <span>{game.mode === 'ranked' ? 'Ranked Match' : 'Casual Match'}</span>
            <span>{game.spectatorCount} watching</span>
          </div>
        </CardHeader>
        <CardContent className="flex flex-col items-center gap-6">
          {/* Game Timers */}
          {game.timeControlMs !== null && (
            <SpectatorTimers
              player1TimeMs={game.player1TimeMs}
              player2TimeMs={game.player2TimeMs}
              turnStartedAt={game.turnStartedAt}
              currentTurn={game.currentTurn}
              player1Name={game.player1.displayName}
              player2Name={game.player2.displayName}
              gameStatus={game.status}
              className="w-full max-w-xs"
            />
          )}

          {game.board && (
            <GameBoard
              board={game.board}
              currentPlayer={game.currentTurn}
              winner={
                game.winner === 'draw'
                  ? 'draw'
                  : game.winner === '1'
                    ? 1
                    : game.winner === '2'
                      ? 2
                      : null
              }
              onColumnClick={() => {}}
              disabled={true}
              threats={[]}
              showThreats={false}
            />
          )}

          <div className="flex flex-col gap-3 w-full">
            <p className="text-center text-sm text-muted-foreground">
              Moves: {game.moves.length}
            </p>

            {/* Chat messages for bot vs bot games */}
            {isBotVsBot && chat.messages.length > 0 && (
              <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg max-h-32 overflow-y-auto">
                <p className="text-xs font-medium text-muted-foreground mb-2">Bot Chat</p>
                <div className="space-y-2">
                  {chat.messages.slice(-5).map((msg) => (
                    <div key={msg.id} className="text-sm">
                      <span className="font-medium text-purple-600 dark:text-purple-400">
                        {msg.senderName}:
                      </span>{' '}
                      <span className="text-muted-foreground">{msg.content}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex gap-2 justify-center">
              <Button onClick={spectator.leaveGame} variant="outline">
                {isGameOver ? 'Back to Games' : 'Leave'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  const renderBotBattleScreen = () => {
    const game = botBattle.game

    // Loading state
    if (botBattle.isLoading) {
      return (
        <Card className="border-purple-500/50">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-2">
              <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.616 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.715-5.349L11 6.477V16h2a1 1 0 110 2H7a1 1 0 110-2h2V6.477L6.237 7.582l1.715 5.349a1 1 0 01-.285 1.05A3.989 3.989 0 015 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.788l1.599.799L9 4.323V3a1 1 0 011-1zm-5 8.274l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L5 10.274zm10 0l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L15 10.274z" />
                </svg>
                Bot Battle
              </span>
            </div>
            <CardTitle className="text-2xl">Generating Match...</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4 py-8">
            <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-sm text-muted-foreground">Finding opponents and playing game...</p>
          </CardContent>
        </Card>
      )
    }

    // Error state
    if (botBattle.status === 'error') {
      return (
        <Card className="border-red-500/50">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl text-red-500">Error</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-center text-muted-foreground">{botBattle.error}</p>
            <div className="flex gap-2 justify-center">
              <Button onClick={() => botBattle.generateGame()} variant="default">
                Try Again
              </Button>
              <Button onClick={handleBackToLive} variant="outline">
                Back to Live Games
              </Button>
            </div>
          </CardContent>
        </Card>
      )
    }

    // No game yet
    if (!game) {
      return (
        <Card className="border-purple-500/50">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl">Bot Battle</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-center text-muted-foreground">
              Watch AI bots battle it out in real-time!
            </p>
            <div className="flex gap-2 justify-center">
              <Button onClick={() => botBattle.generateGame()} className="bg-purple-600 hover:bg-purple-700">
                Generate Battle
              </Button>
              <Button onClick={handleBackToLive} variant="outline">
                Back to Live Games
              </Button>
            </div>
          </CardContent>
        </Card>
      )
    }

    // Game playback
    const getStatusMessage = (): string => {
      if (botBattle.isFinished) {
        if (game.winner === 'draw') return "It's a draw!"
        const winnerName = game.winner === 'bot1' ? game.bot1.name : game.bot2.name
        return `${winnerName} wins!`
      }
      if (botBattle.isPaused) return 'Paused'
      const currentBot = botBattle.currentPlayer === 1 ? game.bot1.name : game.bot2.name
      return `${currentBot}'s turn`
    }

    const getStatusColor = (): string => {
      if (botBattle.isFinished) {
        return game.winner === 'draw' ? 'text-muted-foreground' : 'text-green-600 dark:text-green-400'
      }
      if (botBattle.isPaused) return 'text-yellow-600 dark:text-yellow-400'
      return botBattle.currentPlayer === 1 ? 'text-red-500' : 'text-yellow-500'
    }

    return (
      <Card className="border-purple-500/50">
        <CardHeader className="text-center pb-2">
          {/* Bot Battle Badge */}
          <div className="flex justify-center mb-2">
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 2a1 1 0 011 1v1.323l3.954 1.582 1.599-.8a1 1 0 01.894 1.79l-1.233.616 1.738 5.42a1 1 0 01-.285 1.05A3.989 3.989 0 0115 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.715-5.349L11 6.477V16h2a1 1 0 110 2H7a1 1 0 110-2h2V6.477L6.237 7.582l1.715 5.349a1 1 0 01-.285 1.05A3.989 3.989 0 015 15a3.989 3.989 0 01-2.667-1.019 1 1 0 01-.285-1.05l1.738-5.42-1.233-.617a1 1 0 01.894-1.788l1.599.799L9 4.323V3a1 1 0 011-1zm-5 8.274l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L5 10.274zm10 0l-.818 2.552c.25.112.526.174.818.174.292 0 .569-.062.818-.174L15 10.274z" />
              </svg>
              Bot Battle
            </span>
          </div>

          {/* Player info */}
          <div className="flex justify-center items-center gap-4 mb-2">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500" />
              <div className="text-left">
                <span className="text-sm font-medium">{game.bot1.name}</span>
                <span className="text-xs text-muted-foreground ml-1">({game.bot1.rating})</span>
              </div>
            </div>
            <span className="text-muted-foreground">vs</span>
            <div className="flex items-center gap-2">
              <div className="text-right">
                <span className="text-sm font-medium">{game.bot2.name}</span>
                <span className="text-xs text-muted-foreground ml-1">({game.bot2.rating})</span>
              </div>
              <div className="w-4 h-4 rounded-full bg-yellow-400" />
            </div>
          </div>

          <CardTitle className={`text-2xl ${getStatusColor()}`}>{getStatusMessage()}</CardTitle>
        </CardHeader>

        <CardContent className="flex flex-col items-center gap-4">
          {/* Game Board */}
          {botBattle.board && (
            <GameBoard
              board={botBattle.board}
              currentPlayer={botBattle.currentPlayer}
              winner={
                botBattle.isFinished
                  ? game.winner === 'draw'
                    ? 'draw'
                    : game.winner === 'bot1'
                      ? 1
                      : 2
                  : null
              }
              onColumnClick={() => {}}
              disabled={true}
              threats={[]}
              showThreats={false}
            />
          )}

          {/* Progress bar */}
          <div className="w-full max-w-xs">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
              <span>Move {botBattle.currentMoveIndex + 1} of {botBattle.totalMoves}</span>
              <span>{botBattle.playbackSpeed}x speed</span>
            </div>
            <div className="h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 transition-all duration-300"
                style={{ width: `${botBattle.progress * 100}%` }}
              />
            </div>
          </div>

          {/* Playback controls */}
          <div className="flex items-center gap-2">
            {/* Speed controls */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => botBattle.setSpeed(botBattle.playbackSpeed / 2)}
              disabled={botBattle.playbackSpeed <= 0.25}
            >
              0.5x
            </Button>

            {/* Play/Pause */}
            {botBattle.isPlaying ? (
              <Button variant="outline" size="sm" onClick={botBattle.pause}>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </Button>
            ) : botBattle.isPaused ? (
              <Button variant="outline" size="sm" onClick={botBattle.resume}>
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                </svg>
              </Button>
            ) : null}

            {/* Restart */}
            <Button variant="outline" size="sm" onClick={botBattle.restart}>
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
              </svg>
            </Button>

            {/* Speed controls */}
            <Button
              variant="outline"
              size="sm"
              onClick={() => botBattle.setSpeed(botBattle.playbackSpeed * 2)}
              disabled={botBattle.playbackSpeed >= 4}
            >
              2x
            </Button>
          </div>

          {/* Action buttons */}
          <div className="flex gap-2 mt-2">
            {botBattle.isFinished && (
              <Button onClick={botBattle.watchAnother} className="bg-purple-600 hover:bg-purple-700">
                Watch Another
              </Button>
            )}
            <Button onClick={handleBackToLive} variant="outline">
              Back to Live Games
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  const renderContent = () => {
    // Bot battle mode takes precedence
    if (viewMode === 'bot-battle') {
      return renderBotBattleScreen()
    }

    // Live games mode
    switch (spectator.status) {
      case 'idle':
      case 'loading':
        return (
          <Card>
            <CardHeader className="text-center">
              <CardTitle className="text-2xl">Live Games</CardTitle>
            </CardHeader>
            <CardContent className="flex justify-center py-16">
              <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            </CardContent>
          </Card>
        )
      case 'browsing':
        return renderBrowsingScreen()
      case 'watching':
        return renderWatchingScreen()
      case 'error':
        return (
          <Card>
            <CardHeader className="text-center">
              <CardTitle className="text-2xl text-red-500">Error</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-center text-muted-foreground">{spectator.error}</p>
              <Button onClick={spectator.reset} variant="outline" className="w-full">
                Try Again
              </Button>
            </CardContent>
          </Card>
        )
      default:
        return renderBrowsingScreen()
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-lg mx-auto">{renderContent()}</div>
      </main>
    </div>
  )
}
