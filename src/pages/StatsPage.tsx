import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import {
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts'

interface UserStats {
  user: {
    rating: number
    gamesPlayed: number
    wins: number
    losses: number
    draws: number
  }
  stats: {
    peakRating: number
    lowestRating: number
    currentStreak: number
    longestWinStreak: number
    longestLossStreak: number
    ratingTrend: 'improving' | 'declining' | 'stable'
    recentRatingChange: number
  }
  ratingHistory: Array<{
    rating: number
    createdAt: number
  }>
}

interface StatsHistory {
  dateRange: {
    start: number
    end: number
  }
  summary: {
    totalGames: number
    wins: number
    losses: number
    draws: number
    avgMovesToWin: number
    avgMovesToLoss: number
    player1WinRate: number
    player2WinRate: number
  }
  dailyStats: Array<{
    date: string
    games: number
    wins: number
    losses: number
    draws: number
    ratingChange: number
    avgMoveCount: number
  }>
  weeklyStats: Array<{
    week: string
    games: number
    wins: number
    losses: number
    draws: number
  }>
  winRateOverTime: Array<{
    date: string
    winRate: number
    games: number
  }>
  openingStats: Array<{
    column: number
    games: number
    wins: number
    losses: number
    draws: number
    winRate: number
  }>
  recentGames: Array<{
    id: string
    outcome: string
    opponentType: string
    aiDifficulty: string | null
    playerNumber: number
    moveCount: number
    ratingChange: number
    createdAt: number
    firstMove: number | null
  }>
}

type DateRange = '7d' | '30d' | '90d' | 'all'

const COLORS = {
  wins: '#22c55e',
  losses: '#ef4444',
  draws: '#eab308',
  primary: '#6366f1',
}

const COLUMN_NAMES = ['Far Left', 'Left', 'Mid-Left', 'Center', 'Mid-Right', 'Right', 'Far Right']

export default function StatsPage() {
  const { logout, user } = useAuth()
  const { apiCall } = useAuthenticatedApi()

  const [stats, setStats] = useState<UserStats | null>(null)
  const [history, setHistory] = useState<StatsHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [dateRange, setDateRange] = useState<DateRange>('30d')

  const getDateRangeParams = useCallback((range: DateRange) => {
    const now = Date.now()
    switch (range) {
      case '7d':
        return { start: now - 7 * 24 * 60 * 60 * 1000, end: now }
      case '30d':
        return { start: now - 30 * 24 * 60 * 60 * 1000, end: now }
      case '90d':
        return { start: now - 90 * 24 * 60 * 60 * 1000, end: now }
      case 'all':
        return { start: 0, end: now }
    }
  }, [])

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const { start, end } = getDateRangeParams(dateRange)

        const [statsData, historyData] = await Promise.all([
          apiCall<UserStats>('/api/users/me/stats'),
          apiCall<StatsHistory>(`/api/users/me/stats/history?start=${start}&end=${end}`),
        ])

        setStats(statsData)
        setHistory(historyData)
        setError(null)
      } catch (err) {
        setError('Failed to load statistics')
        console.error('Failed to fetch stats:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [apiCall, dateRange, getDateRangeParams])

  const exportCSV = () => {
    if (!history) return

    const rows = [
      ['Date', 'Games', 'Wins', 'Losses', 'Draws', 'Rating Change', 'Avg Moves'],
      ...history.dailyStats.map((day) => [
        day.date,
        day.games.toString(),
        day.wins.toString(),
        day.losses.toString(),
        day.draws.toString(),
        day.ratingChange.toString(),
        day.avgMoveCount.toString(),
      ]),
    ]

    const csv = rows.map((row) => row.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `makefour-stats-${dateRange}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const winRate = user && user.gamesPlayed > 0 ? Math.round((user.wins / user.gamesPlayed) * 100) : 0

  const pieData = history
    ? [
        { name: 'Wins', value: history.summary.wins, color: COLORS.wins },
        { name: 'Losses', value: history.summary.losses, color: COLORS.losses },
        { name: 'Draws', value: history.summary.draws, color: COLORS.draws },
      ].filter((d) => d.value > 0)
    : []

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">Statistics</h1>
            {user && <p className="text-xs text-muted-foreground">{user.email}</p>}
          </div>
          <div className="flex gap-2">
            <Link to="/dashboard">
              <Button variant="outline" size="sm">
                Dashboard
              </Button>
            </Link>
            <Link to="/profile">
              <Button variant="outline" size="sm">
                Profile
              </Button>
            </Link>
            <ThemeToggle />
            <Button variant="outline" onClick={logout} size="sm">
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {/* Date Range Filter */}
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div className="flex gap-1 bg-muted/50 p-1 rounded-lg">
            {(['7d', '30d', '90d', 'all'] as DateRange[]).map((range) => (
              <button
                key={range}
                type="button"
                onClick={() => setDateRange(range)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  dateRange === range
                    ? 'bg-white dark:bg-gray-800 shadow-sm'
                    : 'hover:bg-white/50 dark:hover:bg-gray-800/50'
                }`}
              >
                {range === '7d' && 'Last 7 Days'}
                {range === '30d' && 'Last 30 Days'}
                {range === '90d' && 'Last 90 Days'}
                {range === 'all' && 'All Time'}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm" onClick={exportCSV} disabled={!history}>
            Export CSV
          </Button>
        </div>

        {loading && (
          <Card>
            <CardContent className="py-8 text-center">
              <div className="animate-pulse">Loading statistics...</div>
            </CardContent>
          </Card>
        )}

        {error && !loading && (
          <Card>
            <CardContent className="py-8 text-center text-red-600 dark:text-red-400">{error}</CardContent>
          </Card>
        )}

        {!loading && !error && stats && history && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-primary">{stats.user.rating}</div>
                    <div className="text-sm text-muted-foreground">Current Rating</div>
                    {stats.stats.ratingTrend !== 'stable' && (
                      <div
                        className={`text-xs mt-1 ${
                          stats.stats.ratingTrend === 'improving'
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}
                      >
                        {stats.stats.ratingTrend === 'improving' ? '↑' : '↓'} {stats.stats.recentRatingChange}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold">{history.summary.totalGames}</div>
                    <div className="text-sm text-muted-foreground">Games Played</div>
                    <div className="text-xs text-muted-foreground mt-1">in selected period</div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600 dark:text-green-400">{winRate}%</div>
                    <div className="text-sm text-muted-foreground">Win Rate</div>
                    <div className="text-xs text-muted-foreground mt-1">overall</div>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <div
                      className={`text-3xl font-bold ${
                        stats.stats.currentStreak >= 0
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-red-600 dark:text-red-400'
                      }`}
                    >
                      {stats.stats.currentStreak >= 0
                        ? `W${stats.stats.currentStreak}`
                        : `L${Math.abs(stats.stats.currentStreak)}`}
                    </div>
                    <div className="text-sm text-muted-foreground">Current Streak</div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Charts Row */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Win/Loss/Draw Pie Chart */}
              <Card>
                <CardHeader>
                  <CardTitle>Game Outcomes</CardTitle>
                  <CardDescription>Win/Loss/Draw distribution</CardDescription>
                </CardHeader>
                <CardContent>
                  {pieData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                        >
                          {pieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend />
                      </PieChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-[250px] flex items-center justify-center text-muted-foreground">
                      No games in selected period
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Win Rate Over Time */}
              <Card>
                <CardHeader>
                  <CardTitle>Win Rate Trend</CardTitle>
                  <CardDescription>Cumulative win percentage over time</CardDescription>
                </CardHeader>
                <CardContent>
                  {history.winRateOverTime.length > 0 ? (
                    <ResponsiveContainer width="100%" height={250}>
                      <LineChart data={history.winRateOverTime}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" tickFormatter={formatDate} className="text-xs" />
                        <YAxis domain={[0, 100]} className="text-xs" />
                        <Tooltip
                          labelFormatter={(label) => formatDate(label as string)}
                          formatter={(value) => [`${value}%`, 'Win Rate']}
                        />
                        <Line
                          type="monotone"
                          dataKey="winRate"
                          stroke={COLORS.primary}
                          strokeWidth={2}
                          dot={false}
                          activeDot={{ r: 4 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-[250px] flex items-center justify-center text-muted-foreground">
                      No games in selected period
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Rating History Chart */}
            {stats.ratingHistory.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Rating History</CardTitle>
                  <CardDescription>
                    Your ELO rating progression (Peak: {stats.stats.peakRating}, Low: {stats.stats.lowestRating})
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart
                      data={stats.ratingHistory.slice(-50).map((entry) => ({
                        date: new Date(entry.createdAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                        rating: entry.rating,
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="date" className="text-xs" />
                      <YAxis
                        domain={['dataMin - 50', 'dataMax + 50']}
                        className="text-xs"
                      />
                      <Tooltip />
                      <Area
                        type="monotone"
                        dataKey="rating"
                        stroke={COLORS.primary}
                        fill={COLORS.primary}
                        fillOpacity={0.2}
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Games Per Day Chart */}
            {history.dailyStats.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Daily Activity</CardTitle>
                  <CardDescription>Games played per day with outcomes</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={history.dailyStats}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis dataKey="date" tickFormatter={formatDate} className="text-xs" />
                      <YAxis className="text-xs" />
                      <Tooltip labelFormatter={(label) => formatDate(label as string)} />
                      <Legend />
                      <Bar dataKey="wins" stackId="a" fill={COLORS.wins} name="Wins" />
                      <Bar dataKey="losses" stackId="a" fill={COLORS.losses} name="Losses" />
                      <Bar dataKey="draws" stackId="a" fill={COLORS.draws} name="Draws" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            )}

            {/* Performance Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Position Stats */}
              <Card>
                <CardHeader>
                  <CardTitle>Performance by Position</CardTitle>
                  <CardDescription>Win rate as Player 1 (Red) vs Player 2 (Yellow)</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Player 1 (Red - First)</span>
                        <span className="text-sm text-muted-foreground">{history.summary.player1WinRate}%</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-3">
                        <div
                          className="bg-red-500 h-3 rounded-full transition-all"
                          style={{ width: `${history.summary.player1WinRate}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium">Player 2 (Yellow - Second)</span>
                        <span className="text-sm text-muted-foreground">{history.summary.player2WinRate}%</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-3">
                        <div
                          className="bg-yellow-500 h-3 rounded-full transition-all"
                          style={{ width: `${history.summary.player2WinRate}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Game Length Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle>Game Length Analysis</CardTitle>
                  <CardDescription>Average moves to win vs loss</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-center">
                      <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                          {history.summary.avgMovesToWin || '-'}
                        </div>
                        <div className="text-sm text-muted-foreground">Avg Moves to Win</div>
                      </div>
                      <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                          {history.summary.avgMovesToLoss || '-'}
                        </div>
                        <div className="text-sm text-muted-foreground">Avg Moves to Loss</div>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground text-center">
                      {history.summary.avgMovesToWin < history.summary.avgMovesToLoss
                        ? 'You tend to win games faster than you lose them - efficient play!'
                        : history.summary.avgMovesToWin > history.summary.avgMovesToLoss
                          ? 'Your losses tend to come quicker - watch for early mistakes'
                          : 'Your win and loss game lengths are similar'}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Opening Column Stats */}
            {history.openingStats.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Opening Column Analysis</CardTitle>
                  <CardDescription>Performance by your first move column</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                    {history.openingStats.slice(0, 8).map((opening) => (
                      <div
                        key={opening.column}
                        className="p-4 bg-muted/50 rounded-lg text-center"
                      >
                        <div className="text-lg font-semibold">{COLUMN_NAMES[opening.column]}</div>
                        <div className="text-xs text-muted-foreground mb-2">Column {opening.column + 1}</div>
                        <div
                          className={`text-2xl font-bold ${
                            opening.winRate >= 50
                              ? 'text-green-600 dark:text-green-400'
                              : 'text-red-600 dark:text-red-400'
                          }`}
                        >
                          {opening.winRate}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {opening.games} games ({opening.wins}W/{opening.losses}L/{opening.draws}D)
                        </div>
                      </div>
                    ))}
                  </div>
                  {history.openingStats.length > 0 && (
                    <div className="mt-4 text-center text-sm text-muted-foreground">
                      Best opening:{' '}
                      <span className="font-medium text-green-600 dark:text-green-400">
                        {COLUMN_NAMES[history.openingStats.reduce((best, curr) =>
                          curr.winRate > best.winRate && curr.games >= 3 ? curr : best
                        ).column]}
                      </span>
                      {' | '}
                      Most used:{' '}
                      <span className="font-medium">{COLUMN_NAMES[history.openingStats[0].column]}</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Streak Records */}
            <Card>
              <CardHeader>
                <CardTitle>Streak Records</CardTitle>
                <CardDescription>Your best and worst streaks</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-center">
                  <div className="p-4 bg-muted/50 rounded-lg">
                    <div
                      className={`text-2xl font-bold ${
                        stats.stats.currentStreak >= 0
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-red-600 dark:text-red-400'
                      }`}
                    >
                      {stats.stats.currentStreak >= 0
                        ? `W${stats.stats.currentStreak}`
                        : `L${Math.abs(stats.stats.currentStreak)}`}
                    </div>
                    <div className="text-sm text-muted-foreground">Current Streak</div>
                  </div>
                  <div className="p-4 bg-muted/50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {stats.stats.longestWinStreak}
                    </div>
                    <div className="text-sm text-muted-foreground">Best Win Streak</div>
                  </div>
                  <div className="p-4 bg-muted/50 rounded-lg">
                    <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                      {stats.stats.longestLossStreak}
                    </div>
                    <div className="text-sm text-muted-foreground">Worst Loss Streak</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Recent Games */}
            {history.recentGames.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recent Form</CardTitle>
                  <CardDescription>Your last {history.recentGames.length} games</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {history.recentGames.map((game) => (
                      <div
                        key={game.id}
                        className={`w-8 h-8 rounded flex items-center justify-center text-white text-sm font-bold ${
                          game.outcome === 'win'
                            ? 'bg-green-500'
                            : game.outcome === 'loss'
                              ? 'bg-red-500'
                              : 'bg-yellow-500'
                        }`}
                        title={`${game.outcome.charAt(0).toUpperCase() + game.outcome.slice(1)} vs ${game.opponentType}${game.aiDifficulty ? ` (${game.aiDifficulty})` : ''} - ${game.moveCount} moves`}
                      >
                        {game.outcome === 'win' ? 'W' : game.outcome === 'loss' ? 'L' : 'D'}
                      </div>
                    ))}
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Result</th>
                          <th className="text-left py-2">Opponent</th>
                          <th className="text-left py-2">Position</th>
                          <th className="text-left py-2">Moves</th>
                          <th className="text-right py-2">Rating</th>
                        </tr>
                      </thead>
                      <tbody>
                        {history.recentGames.map((game) => (
                          <tr key={game.id} className="border-b border-muted">
                            <td className="py-2">
                              <span
                                className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${
                                  game.outcome === 'win'
                                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                                    : game.outcome === 'loss'
                                      ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
                                      : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                                }`}
                              >
                                {game.outcome.toUpperCase()}
                              </span>
                            </td>
                            <td className="py-2 capitalize">
                              {game.opponentType}
                              {game.aiDifficulty && (
                                <span className="text-muted-foreground text-xs ml-1">({game.aiDifficulty})</span>
                              )}
                            </td>
                            <td className="py-2">{game.playerNumber === 1 ? 'Red (1st)' : 'Yellow (2nd)'}</td>
                            <td className="py-2">{game.moveCount}</td>
                            <td
                              className={`py-2 text-right font-medium ${
                                game.ratingChange > 0
                                  ? 'text-green-600 dark:text-green-400'
                                  : game.ratingChange < 0
                                    ? 'text-red-600 dark:text-red-400'
                                    : 'text-muted-foreground'
                              }`}
                            >
                              {game.ratingChange > 0 ? '+' : ''}
                              {game.ratingChange}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Empty State */}
            {history.summary.totalGames === 0 && (
              <Card>
                <CardContent className="py-12 text-center">
                  <div className="text-muted-foreground mb-4">No games found in the selected period.</div>
                  <Link to="/play">
                    <Button>Play a Game</Button>
                  </Link>
                </CardContent>
              </Card>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
