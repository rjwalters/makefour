import { useState, useEffect } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import BotAvatar from '../components/BotAvatar'

interface RecentGame {
  id: string
  outcome: 'win' | 'loss' | 'draw'
  moveCount: number
  ratingChange: number
  opponentType: string
  createdAt: number
}

interface BotProfile {
  id: string
  name: string
  description: string
  avatarUrl: string | null
  playStyle: string
  baseElo: number
  createdAt: number
  rating: number
  gamesPlayed: number
  wins: number
  losses: number
  draws: number
  winRate: number
  recentGames: RecentGame[]
  ratingHistory: Array<{ rating: number; createdAt: number }>
}

export default function BotProfilePage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { isAuthenticated } = useAuth()
  const [profile, setProfile] = useState<BotProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchBotProfile() {
      if (!id) return

      try {
        setLoading(true)
        const response = await fetch(`/api/bot/personas/${id}`)
        if (!response.ok) {
          if (response.status === 404) {
            throw new Error('Bot not found')
          }
          throw new Error('Failed to fetch bot profile')
        }
        const data = await response.json()
        setProfile(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setLoading(false)
      }
    }

    fetchBotProfile()
  }, [id])

  const handleChallenge = () => {
    // Navigate to play page with bot pre-selected
    navigate(`/play?bot=${id}`)
  }

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    })
  }

  const getPlayStyleColor = (style: string) => {
    switch (style) {
      case 'aggressive':
        return 'text-red-600 dark:text-red-400'
      case 'defensive':
        return 'text-blue-600 dark:text-blue-400'
      case 'tricky':
        return 'text-purple-600 dark:text-purple-400'
      case 'adaptive':
        return 'text-green-600 dark:text-green-400'
      default:
        return 'text-gray-600 dark:text-gray-400'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <Link to="/leaderboard">
              <Button variant="ghost" size="sm">
                &larr; Back to Leaderboard
              </Button>
            </Link>
            <div>
              <h1 className="text-2xl font-bold">Bot Profile</h1>
              <p className="text-xs text-muted-foreground">AI Opponent</p>
            </div>
          </div>
          <div className="flex gap-2">
            <ThemeToggle />
            {isAuthenticated && (
              <Link to="/dashboard">
                <Button variant="outline" size="sm">Dashboard</Button>
              </Link>
            )}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {loading && (
          <Card>
            <CardContent className="py-8 text-center">
              <div className="animate-pulse">Loading bot profile...</div>
            </CardContent>
          </Card>
        )}

        {error && !loading && (
          <Card>
            <CardContent className="py-8 text-center text-red-600 dark:text-red-400">
              {error}
            </CardContent>
          </Card>
        )}

        {!loading && !error && profile && (
          <div className="space-y-6">
            {/* Bot Info Card */}
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <BotAvatar
                      avatarUrl={profile.avatarUrl}
                      name={profile.name}
                      size="xl"
                    />
                    <div>
                      <CardTitle className="text-2xl">{profile.name}</CardTitle>
                      <CardDescription className="mt-1">{profile.description}</CardDescription>
                      <div className="flex items-center gap-3 mt-2">
                        <span className={`text-sm font-medium capitalize ${getPlayStyleColor(profile.playStyle)}`}>
                          {profile.playStyle} style
                        </span>
                        <span className="text-sm text-muted-foreground">
                          Base ELO: {profile.baseElo}
                        </span>
                      </div>
                    </div>
                  </div>
                  {isAuthenticated && (
                    <Button onClick={handleChallenge} className="bg-purple-600 hover:bg-purple-700">
                      Challenge
                    </Button>
                  )}
                </div>
              </CardHeader>
            </Card>

            {/* Stats Card */}
            <Card>
              <CardHeader>
                <CardTitle>Performance</CardTitle>
                <CardDescription>Current rating and game statistics</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex flex-wrap gap-6 items-center">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-purple-600 dark:text-purple-400">{profile.rating}</div>
                    <div className="text-sm text-muted-foreground">Rating</div>
                  </div>
                  <div className="h-16 w-px bg-border hidden sm:block" />
                  <div className="flex gap-6 text-center">
                    <div>
                      <div className="text-2xl font-semibold">{profile.gamesPlayed}</div>
                      <div className="text-sm text-muted-foreground">Games</div>
                    </div>
                    <div>
                      <div className="text-2xl font-semibold text-green-600 dark:text-green-400">{profile.wins}</div>
                      <div className="text-sm text-muted-foreground">Wins</div>
                    </div>
                    <div>
                      <div className="text-2xl font-semibold text-red-600 dark:text-red-400">{profile.losses}</div>
                      <div className="text-sm text-muted-foreground">Losses</div>
                    </div>
                    <div>
                      <div className="text-2xl font-semibold text-yellow-600 dark:text-yellow-400">{profile.draws}</div>
                      <div className="text-sm text-muted-foreground">Draws</div>
                    </div>
                    <div>
                      <div className="text-2xl font-semibold">{profile.winRate}%</div>
                      <div className="text-sm text-muted-foreground">Win Rate</div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Rating History Chart */}
            {profile.ratingHistory.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Rating History</CardTitle>
                  <CardDescription>Last {profile.ratingHistory.length} games</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-32 flex items-end gap-1">
                    {profile.ratingHistory.map((entry, index) => {
                      const min = Math.min(...profile.ratingHistory.map(e => e.rating))
                      const max = Math.max(...profile.ratingHistory.map(e => e.rating))
                      const range = max - min || 1
                      const height = ((entry.rating - min) / range) * 100
                      return (
                        <div
                          key={index}
                          className="flex-1 bg-purple-400/60 hover:bg-purple-500 rounded-t transition-colors"
                          style={{ height: `${Math.max(height, 5)}%` }}
                          title={`${entry.rating} - ${formatDate(entry.createdAt)}`}
                        />
                      )
                    })}
                  </div>
                  <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                    <span>Oldest</span>
                    <span>Most Recent</span>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Recent Games */}
            {profile.recentGames.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recent Games</CardTitle>
                  <CardDescription>Last {profile.recentGames.length} games played</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {profile.recentGames.map((game) => (
                      <div
                        key={game.id}
                        className="flex items-center justify-between p-3 bg-muted/50 rounded-lg"
                      >
                        <div className="flex items-center gap-3">
                          <span
                            className={`inline-flex items-center justify-center w-16 px-2 py-1 rounded text-xs font-medium ${
                              game.outcome === 'win'
                                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                                : game.outcome === 'loss'
                                ? 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
                                : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                            }`}
                          >
                            {game.outcome.toUpperCase()}
                          </span>
                          <span className="text-sm text-muted-foreground">
                            vs {game.opponentType}
                          </span>
                        </div>
                        <div className="flex items-center gap-4">
                          <span className="text-sm text-muted-foreground">
                            {game.moveCount} moves
                          </span>
                          <span
                            className={`text-sm font-medium ${
                              game.ratingChange > 0
                                ? 'text-green-600 dark:text-green-400'
                                : game.ratingChange < 0
                                ? 'text-red-600 dark:text-red-400'
                                : 'text-muted-foreground'
                            }`}
                          >
                            {game.ratingChange > 0 ? '+' : ''}{game.ratingChange}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {formatDate(game.createdAt)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {profile.gamesPlayed === 0 && (
              <Card>
                <CardContent className="py-8 text-center text-muted-foreground">
                  This bot hasn't played any games yet. Be the first to challenge them!
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
