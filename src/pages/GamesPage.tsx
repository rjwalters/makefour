import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import Navbar from '../components/Navbar'
import ExportModal from '../components/ExportModal'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'

interface ExportFilters {
  dateFrom?: string
  dateTo?: string
  minMoves?: number
  maxMoves?: number
  outcomes?: ('win' | 'loss' | 'draw')[]
  opponentTypes?: ('human' | 'ai')[]
  aiDifficulties?: ('beginner' | 'intermediate' | 'expert' | 'perfect')[]
  limit?: number
}

interface Game {
  id: string
  outcome: 'win' | 'loss' | 'draw'
  moves: number[]
  moveCount: number
  ratingChange: number
  opponentType: 'human' | 'ai'
  aiDifficulty: 'beginner' | 'intermediate' | 'expert' | 'perfect' | null
  playerNumber: number
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
  const [isExportModalOpen, setIsExportModalOpen] = useState(false)

  const handleExport = async (format: 'json' | 'pgn', filters: ExportFilters) => {
    try {
      const response = await apiCall<{ content?: string; filename?: string; games?: unknown[] }>('/api/export/games', {
        method: 'POST',
        body: JSON.stringify({ format, filters }),
      })

      // Determine filename and content based on format
      let filename: string
      let content: string
      let mimeType: string

      if (format === 'pgn') {
        // PGN response has content and filename fields
        filename = response.filename || `makefour-games-${new Date().toISOString().split('T')[0]}.pgn`
        content = response.content || ''
        mimeType = 'text/plain'
      } else {
        // JSON response is the full export data
        filename = `makefour-games-${new Date().toISOString().split('T')[0]}.json`
        content = JSON.stringify(response, null, 2)
        mimeType = 'application/json'
      }

      // Download the file
      const blob = new Blob([content], { type: mimeType })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
      throw err // Re-throw so ExportModal knows export failed
    }
  }

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

  const getOpponentLabel = (game: Game) => {
    if (game.opponentType === 'human') {
      return 'Hotseat'
    }
    const difficultyLabels: Record<string, string> = {
      beginner: 'Beginner',
      intermediate: 'Intermediate',
      expert: 'Expert',
      perfect: 'Perfect',
    }
    return `vs AI (${difficultyLabels[game.aiDifficulty || 'intermediate']})`
  }

  const getPlayerColorLabel = (game: Game) => {
    if (game.opponentType === 'human') return null
    return game.playerNumber === 1 ? 'Red' : 'Yellow'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        <div className="max-w-2xl mx-auto">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>My Games</CardTitle>
                  <CardDescription>
                    Your game history ({pagination.total} games)
                  </CardDescription>
                </div>
                {pagination.total > 0 && (
                  <Button variant="outline" size="sm" onClick={() => setIsExportModalOpen(true)}>
                    Export
                  </Button>
                )}
              </div>
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
                              {getOpponentLabel(game)}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {game.moveCount} moves
                              {getPlayerColorLabel(game) && (
                                <> · Played as {getPlayerColorLabel(game)}</>
                              )}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {formatDate(game.createdAt)}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          {game.ratingChange !== 0 && (
                            <span
                              className={`text-sm font-medium ${
                                game.ratingChange > 0
                                  ? 'text-green-600 dark:text-green-400'
                                  : 'text-red-600 dark:text-red-400'
                              }`}
                            >
                              {game.ratingChange > 0 ? '+' : ''}
                              {game.ratingChange}
                            </span>
                          )}
                          <Button variant="ghost" size="sm">
                            Replay
                          </Button>
                        </div>
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

      <ExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        onExport={handleExport}
        totalGames={pagination.total}
      />
    </div>
  )
}
