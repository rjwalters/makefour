import { useState, useEffect } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useAuthenticatedApi } from '../hooks/useAuthenticatedApi'
import { usePlayerBotStats, getRecordIndicator } from '../hooks/usePlayerBotStats'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Input } from '../components/ui/input'
import Navbar from '../components/Navbar'

interface UserStats {
  user: {
    id: string
    email: string
    email_verified: boolean
    oauth_provider: string | null
    rating: number
    gamesPlayed: number
    wins: number
    losses: number
    draws: number
    createdAt: number
    lastLogin: number
    updatedAt: number
  }
  stats: {
    peakRating: number
    lowestRating: number
    currentStreak: number
    longestWinStreak: number
    longestLossStreak: number
    avgMoveCount: number
    gamesAsPlayer1: number
    gamesAsPlayer2: number
    aiGames: number
    humanGames: number
    ratingTrend: 'improving' | 'declining' | 'stable'
    recentRatingChange: number
  }
  ratingHistory: Array<{
    rating: number
    createdAt: number
  }>
}

type TabType = 'overview' | 'account' | 'statistics' | 'bot-records' | 'security' | 'settings'

interface RecentGame {
  id: string
  outcome: 'win' | 'loss' | 'draw'
  moveCount: number
  ratingChange: number
  opponentType: 'human' | 'ai'
  aiDifficulty: string | null
  createdAt: number
}

export default function ProfilePage() {
  const { user, logout } = useAuth()
  const { apiCall } = useAuthenticatedApi()
  const navigate = useNavigate()
  const { data: botStatsData, loading: botStatsLoading } = usePlayerBotStats()

  const [activeTab, setActiveTab] = useState<TabType>('overview')
  const [stats, setStats] = useState<UserStats | null>(null)
  const [recentGames, setRecentGames] = useState<RecentGame[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Password change state
  const [oldPassword, setOldPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [passwordError, setPasswordError] = useState<string | null>(null)
  const [passwordSuccess, setPasswordSuccess] = useState(false)
  const [changingPassword, setChangingPassword] = useState(false)

  // Delete account state
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [deleteConfirmText, setDeleteConfirmText] = useState('')
  const [deleting, setDeleting] = useState(false)

  // Username state
  const [usernameStatus, setUsernameStatus] = useState<{
    username: string | null
    displayName: string
    canChange: boolean
    nextChangeAt: number | null
  } | null>(null)
  const [newUsername, setNewUsername] = useState('')
  const [usernameError, setUsernameError] = useState<string | null>(null)
  const [usernameSuccess, setUsernameSuccess] = useState(false)
  const [savingUsername, setSavingUsername] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        const [statsData, gamesData] = await Promise.all([
          apiCall<UserStats>('/api/users/me/stats'),
          apiCall<{ games: RecentGame[] }>('/api/games?limit=5'),
        ])
        setStats(statsData)
        setRecentGames(gamesData.games)
        setError(null)
      } catch (err) {
        setError('Failed to load profile data')
        console.error('Failed to fetch data:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [apiCall])

  // Fetch username status
  useEffect(() => {
    const fetchUsernameStatus = async () => {
      try {
        const data = await apiCall<{
          username: string | null
          displayName: string
          canChange: boolean
          nextChangeAt: number | null
        }>('/api/users/me/username')
        setUsernameStatus(data)
        if (data.username) {
          setNewUsername(data.username)
        }
      } catch (err) {
        console.error('Failed to fetch username status:', err)
      }
    }
    fetchUsernameStatus()
  }, [apiCall])

  const handleSaveUsername = async (e: React.FormEvent) => {
    e.preventDefault()
    setUsernameError(null)
    setUsernameSuccess(false)

    // Client-side validation
    if (!newUsername) {
      setUsernameError('Username is required')
      return
    }
    if (newUsername.length < 3) {
      setUsernameError('Username must be at least 3 characters')
      return
    }
    if (newUsername.length > 20) {
      setUsernameError('Username must be at most 20 characters')
      return
    }
    if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(newUsername)) {
      setUsernameError('Username must start with a letter and contain only letters, numbers, and underscores')
      return
    }

    try {
      setSavingUsername(true)
      const data = await apiCall<{ success: boolean; username: string; nextChangeAt: number }>('/api/users/me/username', {
        method: 'PUT',
        body: JSON.stringify({ username: newUsername }),
      })
      setUsernameSuccess(true)
      setUsernameStatus(prev => prev ? {
        ...prev,
        username: data.username,
        displayName: data.username,
        canChange: false,
        nextChangeAt: data.nextChangeAt,
      } : null)
    } catch (err) {
      setUsernameError(err instanceof Error ? err.message : 'Failed to save username')
    } finally {
      setSavingUsername(false)
    }
  }

  const handleChangePassword = async (e: React.FormEvent) => {
    e.preventDefault()
    setPasswordError(null)
    setPasswordSuccess(false)

    if (newPassword !== confirmPassword) {
      setPasswordError('Passwords do not match')
      return
    }

    if (newPassword.length < 8) {
      setPasswordError('New password must be at least 8 characters')
      return
    }

    try {
      setChangingPassword(true)
      await apiCall('/api/users/me/password', {
        method: 'PUT',
        body: JSON.stringify({
          old_password: oldPassword,
          new_password: newPassword,
        }),
      })
      setPasswordSuccess(true)
      setOldPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err) {
      setPasswordError(err instanceof Error ? err.message : 'Failed to change password')
    } finally {
      setChangingPassword(false)
    }
  }

  const handleDeleteAccount = async () => {
    if (deleteConfirmText !== 'DELETE') return

    try {
      setDeleting(true)
      await apiCall('/api/users/me', {
        method: 'DELETE',
      })
      await logout()
      navigate('/login')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete account')
      setDeleting(false)
    }
  }

  const winRate = user && user.gamesPlayed > 0
    ? Math.round((user.wins / user.gamesPlayed) * 100)
    : 0

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  }

  const formatDateTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const tabs: Array<{ id: TabType; label: string }> = [
    { id: 'overview', label: 'Overview' },
    { id: 'account', label: 'Account' },
    { id: 'statistics', label: 'Statistics' },
    { id: 'bot-records', label: 'Bot Records' },
    { id: 'security', label: 'Security' },
    { id: 'settings', label: 'Settings' },
  ]

  const getOutcomeStyles = (outcome: RecentGame['outcome']) => {
    switch (outcome) {
      case 'win':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'loss':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      case 'draw':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />

      <main className="container mx-auto px-4 py-8">
        {/* Tab Navigation */}
        <div className="flex gap-1 mb-6 bg-muted/50 p-1 rounded-lg w-fit">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white dark:bg-gray-800 shadow-sm'
                  : 'hover:bg-white/50 dark:hover:bg-gray-800/50'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {loading && (
          <Card>
            <CardContent className="py-8 text-center">
              <div className="animate-pulse">Loading profile data...</div>
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

        {!loading && !error && stats && (
          <>
            {/* Overview Tab - Summary Dashboard */}
            {activeTab === 'overview' && (
              <div className="space-y-6">
                {/* Performance Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle>Performance Summary</CardTitle>
                    <CardDescription>Your current rating and game statistics</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-6 items-center">
                      <div className="text-center">
                        <div className="text-4xl font-bold text-primary">{stats.user.rating}</div>
                        <div className="text-sm text-muted-foreground">Rating</div>
                        {stats.stats.ratingTrend !== 'stable' && (
                          <div className={`text-xs ${
                            stats.stats.ratingTrend === 'improving'
                              ? 'text-green-600 dark:text-green-400'
                              : 'text-red-600 dark:text-red-400'
                          }`}>
                            {stats.stats.ratingTrend === 'improving' ? '↑' : '↓'} {stats.stats.ratingTrend}
                          </div>
                        )}
                      </div>
                      <div className="h-16 w-px bg-border hidden sm:block" />
                      <div className="flex gap-6 text-center">
                        <div>
                          <div className="text-2xl font-semibold">{stats.user.gamesPlayed}</div>
                          <div className="text-sm text-muted-foreground">Games</div>
                        </div>
                        <div>
                          <div className="text-2xl font-semibold text-green-600 dark:text-green-400">{stats.user.wins}</div>
                          <div className="text-sm text-muted-foreground">Wins</div>
                        </div>
                        <div>
                          <div className="text-2xl font-semibold text-red-600 dark:text-red-400">{stats.user.losses}</div>
                          <div className="text-sm text-muted-foreground">Losses</div>
                        </div>
                        <div>
                          <div className="text-2xl font-semibold text-yellow-600 dark:text-yellow-400">{stats.user.draws}</div>
                          <div className="text-sm text-muted-foreground">Draws</div>
                        </div>
                        <div>
                          <div className="text-2xl font-semibold">{winRate}%</div>
                          <div className="text-sm text-muted-foreground">Win Rate</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Recent Games */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle>Recent Games</CardTitle>
                        <CardDescription>Your latest matches</CardDescription>
                      </div>
                      <Link to="/games">
                        <Button variant="outline" size="sm">View All</Button>
                      </Link>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {recentGames.length === 0 ? (
                      <div className="text-center py-4 text-muted-foreground">
                        No games yet. <Link to="/play" className="underline">Play your first game!</Link>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {recentGames.map((game) => (
                          <Link key={game.id} to={`/replay/${game.id}`} className="block">
                            <div className="flex items-center justify-between p-3 rounded-lg border hover:bg-accent transition-colors">
                              <div className="flex items-center gap-3">
                                <span className={`px-2 py-1 rounded text-xs font-medium ${getOutcomeStyles(game.outcome)}`}>
                                  {game.outcome.charAt(0).toUpperCase() + game.outcome.slice(1)}
                                </span>
                                <span className="text-sm">
                                  {game.opponentType === 'human' ? 'Hotseat' : `AI${game.aiDifficulty ? ` (${game.aiDifficulty})` : ''}`}
                                </span>
                                <span className="text-xs text-muted-foreground">{game.moveCount} moves</span>
                              </div>
                              {game.ratingChange !== 0 && (
                                <span className={`text-sm font-medium ${
                                  game.ratingChange > 0
                                    ? 'text-green-600 dark:text-green-400'
                                    : 'text-red-600 dark:text-red-400'
                                }`}>
                                  {game.ratingChange > 0 ? '+' : ''}{game.ratingChange}
                                </span>
                              )}
                            </div>
                          </Link>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Quick Links */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card className="hover:shadow-md transition-shadow">
                    <CardContent className="pt-6">
                      <Link to="/stats" className="block">
                        <div className="text-center">
                          <div className="text-lg font-semibold mb-1">Detailed Statistics</div>
                          <p className="text-sm text-muted-foreground mb-3">Charts, trends, and analysis</p>
                          <Button variant="outline" size="sm">View Stats</Button>
                        </div>
                      </Link>
                    </CardContent>
                  </Card>
                  <Card className="hover:shadow-md transition-shadow">
                    <CardContent className="pt-6">
                      <div className="text-center">
                        <div className="text-lg font-semibold mb-1">Bot Records</div>
                        <p className="text-sm text-muted-foreground mb-3">
                          {botStatsData ? `${botStatsData.summary.botsDefeated} bots defeated` : 'Track your progress'}
                        </p>
                        <Button variant="outline" size="sm" onClick={() => setActiveTab('bot-records')}>View Records</Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}

            {/* Account Tab (renamed from Overview) */}
            {activeTab === 'account' && (
              <div className="space-y-6">
                {/* Profile Info Card */}
                <Card>
                  <CardHeader>
                    <CardTitle>Account Information</CardTitle>
                    <CardDescription>Your account details and status</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <span className="block text-sm text-muted-foreground">Email</span>
                        <p className="font-medium">{stats.user.email}</p>
                      </div>
                      <div>
                        <span className="block text-sm text-muted-foreground">Email Status</span>
                        <p className="font-medium">
                          {stats.user.email_verified ? (
                            <span className="text-green-600 dark:text-green-400">Verified</span>
                          ) : (
                            <span className="text-yellow-600 dark:text-yellow-400">Unverified</span>
                          )}
                        </p>
                      </div>
                      <div>
                        <span className="block text-sm text-muted-foreground">Member Since</span>
                        <p className="font-medium">{formatDate(stats.user.createdAt)}</p>
                      </div>
                      <div>
                        <span className="block text-sm text-muted-foreground">Last Login</span>
                        <p className="font-medium">{formatDateTime(stats.user.lastLogin)}</p>
                      </div>
                      {stats.user.oauth_provider && (
                        <div>
                          <span className="block text-sm text-muted-foreground">Linked Account</span>
                          <p className="font-medium capitalize">{stats.user.oauth_provider}</p>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Username Card */}
                <Card>
                  <CardHeader>
                    <CardTitle>Display Name</CardTitle>
                    <CardDescription>
                      Choose a username to display on the leaderboard and in games
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <form onSubmit={handleSaveUsername} className="space-y-4 max-w-md">
                      {usernameError && (
                        <div className="p-3 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-md text-sm">
                          {usernameError}
                        </div>
                      )}
                      {usernameSuccess && (
                        <div className="p-3 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-md text-sm">
                          Username saved successfully!
                        </div>
                      )}
                      <div>
                        <label htmlFor="username" className="block text-sm font-medium mb-1">
                          Username
                        </label>
                        <Input
                          id="username"
                          type="text"
                          value={newUsername}
                          onChange={(e) => setNewUsername(e.target.value)}
                          placeholder={usernameStatus?.displayName || 'Enter username'}
                          disabled={usernameStatus ? !usernameStatus.canChange : false}
                          minLength={3}
                          maxLength={20}
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                          3-20 characters, letters, numbers, and underscores only. Must start with a letter.
                        </p>
                      </div>
                      {usernameStatus && !usernameStatus.canChange && usernameStatus.nextChangeAt && (
                        <p className="text-sm text-muted-foreground">
                          You can change your username again on {formatDate(usernameStatus.nextChangeAt)}
                        </p>
                      )}
                      <Button
                        type="submit"
                        disabled={savingUsername || (usernameStatus ? !usernameStatus.canChange : false)}
                      >
                        {savingUsername ? 'Saving...' : usernameStatus?.username ? 'Change Username' : 'Set Username'}
                      </Button>
                    </form>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Statistics Tab */}
            {activeTab === 'statistics' && (
              <div className="space-y-6">
                {/* Rating Stats */}
                <Card>
                  <CardHeader>
                    <CardTitle>Rating Statistics</CardTitle>
                    <CardDescription>Your rating progression and records</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-primary">{stats.user.rating}</div>
                        <div className="text-sm text-muted-foreground">Current Rating</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.stats.peakRating}</div>
                        <div className="text-sm text-muted-foreground">Peak Rating</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.stats.lowestRating}</div>
                        <div className="text-sm text-muted-foreground">Lowest Rating</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className={`text-2xl font-bold ${
                          stats.stats.recentRatingChange >= 0
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}>
                          {stats.stats.recentRatingChange >= 0 ? '+' : ''}{stats.stats.recentRatingChange}
                        </div>
                        <div className="text-sm text-muted-foreground">Recent Change</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Streak Stats */}
                <Card>
                  <CardHeader>
                    <CardTitle>Streaks</CardTitle>
                    <CardDescription>Your win and loss streaks</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className={`text-2xl font-bold ${
                          stats.stats.currentStreak >= 0
                            ? 'text-green-600 dark:text-green-400'
                            : 'text-red-600 dark:text-red-400'
                        }`}>
                          {stats.stats.currentStreak >= 0 ? `W${stats.stats.currentStreak}` : `L${Math.abs(stats.stats.currentStreak)}`}
                        </div>
                        <div className="text-sm text-muted-foreground">Current Streak</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">{stats.stats.longestWinStreak}</div>
                        <div className="text-sm text-muted-foreground">Best Win Streak</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">{stats.stats.longestLossStreak}</div>
                        <div className="text-sm text-muted-foreground">Worst Loss Streak</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Game Breakdown */}
                <Card>
                  <CardHeader>
                    <CardTitle>Game Breakdown</CardTitle>
                    <CardDescription>Analysis of your games</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold">{stats.stats.aiGames}</div>
                        <div className="text-sm text-muted-foreground">vs AI</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold">{stats.stats.humanGames}</div>
                        <div className="text-sm text-muted-foreground">vs Human</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold">{stats.stats.gamesAsPlayer1}</div>
                        <div className="text-sm text-muted-foreground">As Red (1st)</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold">{stats.stats.gamesAsPlayer2}</div>
                        <div className="text-sm text-muted-foreground">As Yellow (2nd)</div>
                      </div>
                    </div>
                    {stats.stats.avgMoveCount > 0 && (
                      <div className="mt-4 text-center text-sm text-muted-foreground">
                        Average game length: <span className="font-medium">{stats.stats.avgMoveCount.toFixed(1)} moves</span>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Rating History Chart (Simple visualization) */}
                {stats.ratingHistory.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Rating History</CardTitle>
                      <CardDescription>Your last {stats.ratingHistory.length} games</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-32 flex items-end gap-1">
                        {stats.ratingHistory.slice(-20).map((entry, index) => {
                          const min = Math.min(...stats.ratingHistory.map(e => e.rating))
                          const max = Math.max(...stats.ratingHistory.map(e => e.rating))
                          const range = max - min || 1
                          const height = ((entry.rating - min) / range) * 100
                          return (
                            <div
                              key={index}
                              className="flex-1 bg-primary/60 hover:bg-primary rounded-t transition-colors"
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
              </div>
            )}

            {/* Bot Records Tab */}
            {activeTab === 'bot-records' && (
              <div className="space-y-6">
                {/* Summary Card */}
                {botStatsLoading ? (
                  <Card>
                    <CardContent className="py-8 text-center">
                      <div className="animate-pulse">Loading bot records...</div>
                    </CardContent>
                  </Card>
                ) : botStatsData ? (
                  <>
                    <Card>
                      <CardHeader>
                        <CardTitle>Bot Challenge Progress</CardTitle>
                        <CardDescription>Your overall performance against AI opponents</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center p-4 bg-muted/50 rounded-lg">
                            <div className="text-2xl font-bold">{botStatsData.summary.totalGames}</div>
                            <div className="text-sm text-muted-foreground">Total Bot Games</div>
                          </div>
                          <div className="text-center p-4 bg-muted/50 rounded-lg">
                            <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                              {botStatsData.summary.overallWinRate}%
                            </div>
                            <div className="text-sm text-muted-foreground">Win Rate</div>
                          </div>
                          <div className="text-center p-4 bg-muted/50 rounded-lg">
                            <div className="text-2xl font-bold text-primary">
                              {botStatsData.summary.botsDefeated}/{botStatsData.summary.botsPlayed + botStatsData.summary.botsRemaining}
                            </div>
                            <div className="text-sm text-muted-foreground">Bots Defeated</div>
                          </div>
                          <div className="text-center p-4 bg-muted/50 rounded-lg">
                            <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">
                              {botStatsData.summary.botsMastered}
                            </div>
                            <div className="text-sm text-muted-foreground">Bots Mastered</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    {/* Bot Records Grid */}
                    <Card>
                      <CardHeader>
                        <CardTitle>Individual Bot Records</CardTitle>
                        <CardDescription>
                          Your win/loss record against each bot opponent
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                          {/* Played bots first */}
                          {botStatsData.stats.map((bot) => (
                            <div
                              key={bot.botId}
                              className="p-4 border rounded-lg hover:bg-muted/30 transition-colors"
                            >
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <span className="text-lg">{getRecordIndicator(bot.wins, bot.losses)}</span>
                                  <span className="font-medium">{bot.botName}</span>
                                </div>
                                <span className="text-sm text-muted-foreground">
                                  {bot.botRating} ELO
                                </span>
                              </div>
                              <div className="flex items-center justify-between">
                                <div className="text-lg font-bold">
                                  <span className="text-green-600 dark:text-green-400">{bot.wins}</span>
                                  <span className="text-muted-foreground">-</span>
                                  <span className="text-red-600 dark:text-red-400">{bot.losses}</span>
                                  {bot.draws > 0 && (
                                    <>
                                      <span className="text-muted-foreground">-</span>
                                      <span className="text-yellow-600 dark:text-yellow-400">{bot.draws}</span>
                                    </>
                                  )}
                                </div>
                                <span className="text-sm text-muted-foreground">
                                  {bot.winRate}% WR
                                </span>
                              </div>
                              <div className="mt-2 flex flex-wrap gap-1">
                                {bot.currentStreak !== 0 && (
                                  <span className={`text-xs px-2 py-0.5 rounded ${
                                    bot.currentStreak > 0
                                      ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                                      : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
                                  }`}>
                                    {bot.currentStreak > 0 ? `${bot.currentStreak}W streak` : `${Math.abs(bot.currentStreak)}L streak`}
                                  </span>
                                )}
                                {bot.isMastered && (
                                  <span className="text-xs px-2 py-0.5 rounded bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300">
                                    Mastered
                                  </span>
                                )}
                                {bot.isUndefeated && (
                                  <span className="text-xs px-2 py-0.5 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                                    Undefeated
                                  </span>
                                )}
                              </div>
                            </div>
                          ))}
                          {/* Unplayed bots */}
                          {botStatsData.unplayed.map((bot) => (
                            <div
                              key={bot.botId}
                              className="p-4 border rounded-lg opacity-60 hover:opacity-100 transition-opacity"
                            >
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <span className="text-lg">❓</span>
                                  <span className="font-medium">{bot.botName}</span>
                                </div>
                                <span className="text-sm text-muted-foreground">
                                  {bot.botRating} ELO
                                </span>
                              </div>
                              <div className="text-sm text-muted-foreground italic">
                                Not yet challenged
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </>
                ) : (
                  <Card>
                    <CardContent className="py-8 text-center text-muted-foreground">
                      No bot records found. Play some ranked bot games to start tracking your progress!
                    </CardContent>
                  </Card>
                )}
              </div>
            )}

            {/* Security Tab */}
            {activeTab === 'security' && (
              <div className="space-y-6">
                {/* Change Password Card */}
                <Card>
                  <CardHeader>
                    <CardTitle>Change Password</CardTitle>
                    <CardDescription>
                      {stats.user.oauth_provider
                        ? `You signed in with ${stats.user.oauth_provider}. Password change is not available for OAuth accounts.`
                        : 'Update your password to keep your account secure'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {stats.user.oauth_provider ? (
                      <p className="text-muted-foreground">
                        Your account is linked to {stats.user.oauth_provider}. Manage your password through your {stats.user.oauth_provider} account settings.
                      </p>
                    ) : (
                      <form onSubmit={handleChangePassword} className="space-y-4 max-w-md">
                        {passwordError && (
                          <div className="p-3 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded-md text-sm">
                            {passwordError}
                          </div>
                        )}
                        {passwordSuccess && (
                          <div className="p-3 bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400 rounded-md text-sm">
                            Password changed successfully!
                          </div>
                        )}
                        <div>
                          <label htmlFor="current-password" className="block text-sm font-medium mb-1">Current Password</label>
                          <Input
                            id="current-password"
                            type="password"
                            value={oldPassword}
                            onChange={(e) => setOldPassword(e.target.value)}
                            required
                          />
                        </div>
                        <div>
                          <label htmlFor="new-password" className="block text-sm font-medium mb-1">New Password</label>
                          <Input
                            id="new-password"
                            type="password"
                            value={newPassword}
                            onChange={(e) => setNewPassword(e.target.value)}
                            required
                            minLength={8}
                          />
                        </div>
                        <div>
                          <label htmlFor="confirm-password" className="block text-sm font-medium mb-1">Confirm New Password</label>
                          <Input
                            id="confirm-password"
                            type="password"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            required
                          />
                        </div>
                        <Button type="submit" disabled={changingPassword}>
                          {changingPassword ? 'Changing...' : 'Change Password'}
                        </Button>
                      </form>
                    )}
                  </CardContent>
                </Card>

                {/* Linked Accounts Card */}
                <Card>
                  <CardHeader>
                    <CardTitle>Linked Accounts</CardTitle>
                    <CardDescription>External accounts connected to your profile</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-white rounded-full flex items-center justify-center shadow-sm">
                          <svg className="w-5 h-5" viewBox="0 0 24 24" aria-labelledby="google-icon-title">
                            <title id="google-icon-title">Google</title>
                            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                          </svg>
                        </div>
                        <div>
                          <p className="font-medium">Google</p>
                          <p className="text-sm text-muted-foreground">
                            {stats.user.oauth_provider === 'google' ? 'Connected' : 'Not connected'}
                          </p>
                        </div>
                      </div>
                      <Button variant="outline" size="sm" disabled>
                        {stats.user.oauth_provider === 'google' ? 'Connected' : 'Connect'}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Settings Tab */}
            {activeTab === 'settings' && (
              <div className="space-y-6">
                {/* Danger Zone */}
                <Card className="border-red-200 dark:border-red-800">
                  <CardHeader>
                    <CardTitle className="text-red-600 dark:text-red-400">Danger Zone</CardTitle>
                    <CardDescription>Irreversible actions for your account</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="p-4 border border-red-200 dark:border-red-800 rounded-lg">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                        <div>
                          <h4 className="font-medium">Delete Account</h4>
                          <p className="text-sm text-muted-foreground">
                            Permanently delete your account and all associated data. This action cannot be undone.
                          </p>
                        </div>
                        <Button
                          variant="destructive"
                          onClick={() => setShowDeleteConfirm(true)}
                        >
                          Delete Account
                        </Button>
                      </div>
                    </div>

                    {showDeleteConfirm && (
                      <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                        <h4 className="font-medium text-red-600 dark:text-red-400 mb-2">
                          Are you absolutely sure?
                        </h4>
                        <p className="text-sm text-muted-foreground mb-4">
                          This will permanently delete your account, all your games, statistics, and rating history.
                          This action cannot be undone.
                        </p>
                        <div className="space-y-3">
                          <div>
                            <label htmlFor="delete-confirm" className="block text-sm font-medium mb-1">
                              Type DELETE to confirm
                            </label>
                            <Input
                              id="delete-confirm"
                              value={deleteConfirmText}
                              onChange={(e) => setDeleteConfirmText(e.target.value)}
                              placeholder="DELETE"
                              className="max-w-xs"
                            />
                          </div>
                          <div className="flex gap-2">
                            <Button
                              variant="destructive"
                              onClick={handleDeleteAccount}
                              disabled={deleteConfirmText !== 'DELETE' || deleting}
                            >
                              {deleting ? 'Deleting...' : 'Delete My Account'}
                            </Button>
                            <Button
                              variant="outline"
                              onClick={() => {
                                setShowDeleteConfirm(false)
                                setDeleteConfirmText('')
                              }}
                            >
                              Cancel
                            </Button>
                          </div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}
