import { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import Navbar from '../components/Navbar'
import BotAvatar from '../components/BotAvatar'

interface RecentGame {
  id: string
  outcome: 'win' | 'loss' | 'draw'
  moveCount: number
  ratingChange: number
  opponentType: string
  createdAt: number
}

interface BotMatchup {
  opponentId: string
  opponentName: string
  opponentAvatarUrl: string | null
  totalGames: number
  wins: number
  losses: number
  draws: number
  winRate: number
  avgMoves: number
  lastGameAt: number
}

interface AIConfig {
  searchDepth?: number
  errorRate?: number
  timeMultiplier?: number
  temperature?: number
  useHybridSearch?: boolean
  hybridDepth?: number
  modelId?: string
}

interface BotProfile {
  id: string
  name: string
  description: string
  avatarUrl: string | null
  playStyle: string
  aiEngine?: string
  aiConfig?: AIConfig
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
  matchups?: BotMatchup[]
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

  const getEngineDescription = (engine: string, config?: AIConfig) => {
    if (engine === 'neural') {
      const modelName = config?.modelId || 'pattern-based'
      const temp = config?.temperature ?? 0.5
      if (config?.useHybridSearch) {
        return `Neural network (${modelName}) with ${config.hybridDepth}-ply lookahead. Uses learned patterns combined with tactical search.`
      }
      return `Neural network (${modelName}) with temperature ${temp}. Plays by intuition using learned patterns.`
    }
    // Minimax engine
    const depth = config?.searchDepth ?? 6
    const errorRate = config?.errorRate ?? 0
    const errorPct = Math.round(errorRate * 100)
    if (errorPct > 0) {
      return `Minimax search to depth ${depth} with ${errorPct}% chance of suboptimal moves. Uses alpha-beta pruning for efficient lookahead.`
    }
    return `Minimax search to depth ${depth} with alpha-beta pruning. Evaluates thousands of positions to find optimal moves.`
  }

  const getEngineIcon = (engine: string) => {
    if (engine === 'neural') {
      return (
        <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
      )
    }
    return (
      <svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19.14 12.94c.04-.31.06-.63.06-.94 0-.31-.02-.63-.06-.94l2.03-1.58c.18-.14.23-.41.12-.61l-1.92-3.32c-.12-.22-.37-.29-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54c-.04-.24-.24-.41-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96c-.22-.08-.47 0-.59.22L2.74 8.87c-.12.21-.08.47.12.61l2.03 1.58c-.04.31-.06.63-.06.94s.02.63.06.94l-2.03 1.58c-.18.14-.23.41-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32c.12-.22.07-.47-.12-.61l-2.01-1.58zM12 15.6c-1.98 0-3.6-1.62-3.6-3.6s1.62-3.6 3.6-3.6 3.6 1.62 3.6 3.6-1.62 3.6-3.6 3.6z"/>
      </svg>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />

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

            {/* Under the Hood Card */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {getEngineIcon(profile.aiEngine || 'minimax')}
                  Under the Hood
                </CardTitle>
                <CardDescription>How {profile.name} thinks</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    profile.aiEngine === 'neural'
                      ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200'
                      : 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                  }`}>
                    {profile.aiEngine === 'neural' ? 'Neural Network' : 'Minimax Engine'}
                  </div>
                </div>
                <p className="text-muted-foreground">
                  {getEngineDescription(profile.aiEngine || 'minimax', profile.aiConfig)}
                </p>
                {profile.aiConfig && (
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 pt-2">
                    {profile.aiConfig.searchDepth !== undefined && (
                      <div className="bg-muted/50 rounded-lg p-3">
                        <div className="text-lg font-semibold">{profile.aiConfig.searchDepth}</div>
                        <div className="text-xs text-muted-foreground">Search Depth</div>
                      </div>
                    )}
                    {profile.aiConfig.errorRate !== undefined && profile.aiConfig.errorRate > 0 && (
                      <div className="bg-muted/50 rounded-lg p-3">
                        <div className="text-lg font-semibold">{Math.round(profile.aiConfig.errorRate * 100)}%</div>
                        <div className="text-xs text-muted-foreground">Error Rate</div>
                      </div>
                    )}
                    {profile.aiConfig.temperature !== undefined && (
                      <div className="bg-muted/50 rounded-lg p-3">
                        <div className="text-lg font-semibold">{profile.aiConfig.temperature}</div>
                        <div className="text-xs text-muted-foreground">Temperature</div>
                      </div>
                    )}
                    {profile.aiConfig.hybridDepth !== undefined && (
                      <div className="bg-muted/50 rounded-lg p-3">
                        <div className="text-lg font-semibold">{profile.aiConfig.hybridDepth}</div>
                        <div className="text-xs text-muted-foreground">Hybrid Depth</div>
                      </div>
                    )}
                  </div>
                )}
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

            {/* Bot vs Bot Matchups */}
            {profile.matchups && profile.matchups.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Bot Matchups</CardTitle>
                  <CardDescription>Head-to-head records against other bots</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {profile.matchups.map((matchup) => (
                      <Link
                        key={matchup.opponentId}
                        to={`/bot/${matchup.opponentId}`}
                        className="flex items-center justify-between p-3 bg-muted/50 rounded-lg hover:bg-muted/70 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <BotAvatar
                            avatarUrl={matchup.opponentAvatarUrl}
                            name={matchup.opponentName}
                            size="sm"
                          />
                          <div>
                            <span className="font-medium">vs {matchup.opponentName}</span>
                            <div className="text-xs text-muted-foreground">
                              {matchup.totalGames} games, avg {matchup.avgMoves} moves
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-4">
                          <div className="text-right">
                            <div className="flex items-center gap-2">
                              <span className="text-green-600 dark:text-green-400 font-medium">{matchup.wins}</span>
                              <span className="text-muted-foreground">-</span>
                              <span className="text-red-600 dark:text-red-400 font-medium">{matchup.losses}</span>
                              {matchup.draws > 0 && (
                                <>
                                  <span className="text-muted-foreground">-</span>
                                  <span className="text-yellow-600 dark:text-yellow-400 font-medium">{matchup.draws}</span>
                                </>
                              )}
                            </div>
                          </div>
                          <div className="w-20">
                            <div className="flex items-center gap-2">
                              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${
                                    matchup.winRate >= 60 ? 'bg-green-500' :
                                    matchup.winRate >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                                  }`}
                                  style={{ width: `${matchup.winRate}%` }}
                                />
                              </div>
                              <span className={`text-xs font-medium min-w-[2.5rem] text-right ${
                                matchup.winRate >= 60 ? 'text-green-600 dark:text-green-400' :
                                matchup.winRate >= 40 ? 'text-yellow-600 dark:text-yellow-400' : 'text-red-600 dark:text-red-400'
                              }`}>
                                {matchup.winRate}%
                              </span>
                            </div>
                          </div>
                        </div>
                      </Link>
                    ))}
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
