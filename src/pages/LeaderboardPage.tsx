import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import BotAvatar from '../components/BotAvatar'

interface LeaderboardEntry {
  rank: number
  userId: string
  username: string
  rating: number
  gamesPlayed: number
  wins: number
  losses: number
  draws: number
  winRate: number
  isBot: boolean
  botPersonaId: string | null
  botDescription: string | null
  botAvatarUrl: string | null
}

interface LeaderboardResponse {
  leaderboard: LeaderboardEntry[]
  pagination: {
    total: number
    limit: number
    offset: number
    hasMore: boolean
  }
}

export default function LeaderboardPage() {
  const { logout, user } = useAuth()
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [includeBots, setIncludeBots] = useState(true)
  const [pagination, setPagination] = useState({
    total: 0,
    limit: 50,
    offset: 0,
    hasMore: false,
  })

  useEffect(() => {
    async function fetchLeaderboard() {
      try {
        setLoading(true)
        const response = await fetch(
          `/api/leaderboard?limit=${pagination.limit}&offset=${pagination.offset}&includeBots=${includeBots}`
        )
        if (!response.ok) {
          throw new Error('Failed to fetch leaderboard')
        }
        const data: LeaderboardResponse = await response.json()
        setLeaderboard(data.leaderboard)
        setPagination(data.pagination)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred')
      } finally {
        setLoading(false)
      }
    }

    fetchLeaderboard()
  }, [pagination.offset, pagination.limit, includeBots])

  const toggleBots = () => {
    setIncludeBots(!includeBots)
    // Reset to first page when toggling
    setPagination(prev => ({ ...prev, offset: 0 }))
  }

  const loadMore = () => {
    setPagination((prev) => ({
      ...prev,
      offset: prev.offset + prev.limit,
    }))
  }

  const loadPrevious = () => {
    setPagination((prev) => ({
      ...prev,
      offset: Math.max(0, prev.offset - prev.limit),
    }))
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div className="flex items-center gap-4">
            <Link to="/dashboard">
              <Button variant="ghost" size="sm">
                &larr; Back
              </Button>
            </Link>
            <div>
              <h1 className="text-2xl font-bold">Leaderboard</h1>
              <p className="text-xs text-muted-foreground">Top Players</p>
            </div>
          </div>
          <div className="flex gap-2">
            <ThemeToggle />
            <Button variant="outline" onClick={logout} size="sm">
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle>Rankings</CardTitle>
            <label className="flex items-center gap-2 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={includeBots}
                onChange={toggleBots}
                className="rounded border-gray-300 text-primary focus:ring-primary"
              />
              <span className="text-muted-foreground">Show bots</span>
            </label>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-8 text-muted-foreground">
                Loading leaderboard...
              </div>
            ) : error ? (
              <div className="text-center py-8 text-red-500">{error}</div>
            ) : leaderboard.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                No players have completed any games yet. Be the first!
              </div>
            ) : (
              <>
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-3 px-2 font-medium">Rank</th>
                        <th className="text-left py-3 px-2 font-medium">Player</th>
                        <th className="text-right py-3 px-2 font-medium">Rating</th>
                        <th className="text-right py-3 px-2 font-medium hidden sm:table-cell">Games</th>
                        <th className="text-right py-3 px-2 font-medium hidden md:table-cell">W/L/D</th>
                        <th className="text-right py-3 px-2 font-medium">Win %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {leaderboard.map((entry) => (
                        <tr
                          key={entry.userId}
                          className={`border-b last:border-b-0 hover:bg-muted/50 ${
                            user?.id === entry.userId ? 'bg-primary/10' : ''
                          } ${entry.isBot ? 'bg-purple-50/50 dark:bg-purple-900/10' : ''}`}
                        >
                          <td className="py-3 px-2">
                            <span
                              className={`inline-flex items-center justify-center w-8 h-8 rounded-full ${
                                entry.rank === 1
                                  ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
                                  : entry.rank === 2
                                  ? 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
                                  : entry.rank === 3
                                  ? 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200'
                                  : 'text-muted-foreground'
                              }`}
                            >
                              {entry.rank}
                            </span>
                          </td>
                          <td className="py-3 px-2 font-medium">
                            <div className="flex items-center gap-2">
                              {entry.isBot && (
                                <BotAvatar
                                  avatarUrl={entry.botAvatarUrl}
                                  name={entry.username}
                                  size="xs"
                                />
                              )}
                              <Link
                                to={entry.isBot ? `/bot/${entry.botPersonaId}` : '#'}
                                className={entry.isBot ? 'hover:underline text-purple-700 dark:text-purple-300' : ''}
                              >
                                {entry.username}
                              </Link>
                              {user?.id === entry.userId && (
                                <span className="text-xs text-primary">(You)</span>
                              )}
                            </div>
                          </td>
                          <td className="py-3 px-2 text-right font-bold text-primary">
                            {entry.rating}
                          </td>
                          <td className="py-3 px-2 text-right text-muted-foreground hidden sm:table-cell">
                            {entry.gamesPlayed}
                          </td>
                          <td className="py-3 px-2 text-right hidden md:table-cell">
                            <span className="text-green-600 dark:text-green-400">{entry.wins}</span>
                            <span className="text-muted-foreground">/</span>
                            <span className="text-red-600 dark:text-red-400">{entry.losses}</span>
                            <span className="text-muted-foreground">/</span>
                            <span className="text-yellow-600 dark:text-yellow-400">{entry.draws}</span>
                          </td>
                          <td className="py-3 px-2 text-right">{entry.winRate}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Pagination */}
                <div className="flex justify-between items-center mt-4 pt-4 border-t">
                  <div className="text-sm text-muted-foreground">
                    Showing {pagination.offset + 1}-
                    {Math.min(pagination.offset + leaderboard.length, pagination.total)} of{' '}
                    {pagination.total} players
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={loadPrevious}
                      disabled={pagination.offset === 0}
                    >
                      Previous
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={loadMore}
                      disabled={!pagination.hasMore}
                    >
                      Next
                    </Button>
                  </div>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </main>
    </div>
  )
}
