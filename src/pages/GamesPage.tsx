import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'

interface Game {
  id: string
  outcome: 'win' | 'loss' | 'draw'
  moves: number[]
  moveCount: number
  createdAt: number
}

interface GamesResponse {
  games: Game[]
  pagination: {
    total: number
    limit: number
    offset: number
    hasMore: boolean
  }
}

export default function GamesPage() {
  const { logout, user } = useAuth()
  const { apiCall } = useAuthenticatedApi()
  const [games, setGames] = useState<Game[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [pagination, setPagination] = useState({
    total: 0,
    limit: 20,
    offset: 0,
    hasMore: false,
  })

  const fetchGames = async (offset = 0) => {
    setIsLoading(true)
    setError(null)

    try {
      const data = await apiCall<GamesResponse>(`/api/games?limit=20&offset=${offset}`)
      if (offset === 0) {
        setGames(data.games)
      } else {
        setGames((prev) => [...prev, ...data.games])
      }
      setPagination(data.pagination)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load games')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchGames()
  }, [])

  const loadMore = () => {
    fetchGames(pagination.offset + pagination.limit)
  }

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getOutcomeStyles = (outcome: Game['outcome']) => {
    switch (outcome) {
      case 'win':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'loss':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      case 'draw':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
    }
  }

  const getOutcomeLabel = (outcome: Game['outcome']) => {
    switch (outcome) {
      case 'win':
        return 'Win'
      case 'loss':
        return 'Loss'
      case 'draw':
        return 'Draw'
    }
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
        <div className="max-w-2xl mx-auto">
          <Card>
            <CardHeader>
              <CardTitle>My Games</CardTitle>
              <CardDescription>
                Your game history ({pagination.total} games)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading && games.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  Loading games...
                </div>
              ) : error ? (
                <div className="text-center py-8 text-red-500">{error}</div>
              ) : games.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground mb-4">No games yet</p>
                  <Link to="/play">
                    <Button>Play Your First Game</Button>
                  </Link>
                </div>
              ) : (
                <div className="space-y-3">
                  {games.map((game) => (
                    <Link
                      key={game.id}
                      to={`/replay/${game.id}`}
                      className="block"
                    >
                      <div className="flex items-center justify-between p-4 rounded-lg border hover:bg-accent transition-colors">
                        <div className="flex items-center gap-4">
                          <span
                            className={`px-3 py-1 rounded-full text-sm font-medium ${getOutcomeStyles(
                              game.outcome
                            )}`}
                          >
                            {getOutcomeLabel(game.outcome)}
                          </span>
                          <div>
                            <p className="text-sm font-medium">
                              {game.moveCount} moves
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {formatDate(game.createdAt)}
                            </p>
                          </div>
                        </div>
                        <Button variant="ghost" size="sm">
                          Replay
                        </Button>
                      </div>
                    </Link>
                  ))}

                  {pagination.hasMore && (
                    <div className="pt-4 text-center">
                      <Button
                        variant="outline"
                        onClick={loadMore}
                        disabled={isLoading}
                      >
                        {isLoading ? 'Loading...' : 'Load More'}
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
