import { useCallback } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import GameBoard from '../components/GameBoard'
import { SpectatorTimers } from '../components/GameTimer'
import { useSpectate, type LiveGame } from '../hooks/useSpectate'

export default function SpectatorPage() {
  const { logout, user, isAuthenticated } = useAuth()
  const spectator = useSpectate()

  // Start browsing when page loads (if not already)
  const handleStartBrowsing = useCallback(() => {
    if (spectator.status === 'idle') {
      spectator.startBrowsing()
    }
  }, [spectator])

  const renderIdleScreen = () => (
    <Card>
      <CardHeader className="text-center">
        <CardTitle className="text-2xl">Watch Live Games</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <p className="text-center text-muted-foreground">
          Watch ongoing Connect Four matches between players in real-time.
        </p>
        <Button onClick={handleStartBrowsing} size="lg" className="w-full">
          Browse Live Games
        </Button>
        <Link to="/play" className="block">
          <Button variant="outline" className="w-full">
            Back to Play
          </Button>
        </Link>
      </CardContent>
    </Card>
  )

  const renderGameCard = (game: LiveGame) => {
    const avgRating = Math.round((game.player1.rating + game.player2.rating) / 2)
    const duration = Math.floor((Date.now() - game.createdAt) / 60000)

    return (
      <Card
        key={game.id}
        className="cursor-pointer hover:border-primary transition-colors"
        onClick={() => spectator.watchGame(game.id)}
      >
        <CardContent className="p-4">
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
              {game.currentTurn === 1 ? 'Red' : 'Yellow'}'s turn
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

        <Button onClick={spectator.stopBrowsing} variant="outline" className="w-full">
          Stop Browsing
        </Button>
      </CardContent>
    </Card>
  )

  const renderWatchingScreen = () => {
    const game = spectator.currentGame
    if (!game) return null

    const isGameOver = game.status !== 'active'

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
      <Card>
        <CardHeader className="text-center pb-2">
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

  const renderContent = () => {
    switch (spectator.status) {
      case 'idle':
        return renderIdleScreen()
      case 'browsing':
        return renderBrowsingScreen()
      case 'loading':
        return (
          <Card>
            <CardContent className="flex justify-center py-16">
              <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
            </CardContent>
          </Card>
        )
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
        return renderIdleScreen()
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
        <div className="max-w-lg mx-auto">{renderContent()}</div>
      </main>
    </div>
  )
}
