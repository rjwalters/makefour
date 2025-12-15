import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import Navbar from '../components/Navbar'
import { VerificationBanner } from '../components/VerificationBanner'

export default function DashboardPage() {
  const { user } = useAuth()

  // Calculate win rate
  const winRate = user && user.gamesPlayed > 0
    ? Math.round((user.wins / user.gamesPlayed) * 100)
    : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <Navbar />

      <VerificationBanner />

      <main className="container mx-auto px-4 py-8">
        {/* Rating Stats Card */}
        {user && (
          <Card className="mb-6">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg">Your Stats</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-6 items-center">
                <div className="text-center">
                  <div className="text-3xl font-bold text-primary">{user.rating}</div>
                  <div className="text-xs text-muted-foreground">Rating</div>
                </div>
                <div className="h-12 w-px bg-border hidden sm:block" />
                <div className="flex gap-4 text-center">
                  <div>
                    <div className="text-xl font-semibold">{user.gamesPlayed}</div>
                    <div className="text-xs text-muted-foreground">Games</div>
                  </div>
                  <div>
                    <div className="text-xl font-semibold text-green-600 dark:text-green-400">{user.wins}</div>
                    <div className="text-xs text-muted-foreground">Wins</div>
                  </div>
                  <div>
                    <div className="text-xl font-semibold text-red-600 dark:text-red-400">{user.losses}</div>
                    <div className="text-xs text-muted-foreground">Losses</div>
                  </div>
                  <div>
                    <div className="text-xl font-semibold text-yellow-600 dark:text-yellow-400">{user.draws}</div>
                    <div className="text-xs text-muted-foreground">Draws</div>
                  </div>
                  <div>
                    <div className="text-xl font-semibold">{winRate}%</div>
                    <div className="text-xs text-muted-foreground">Win Rate</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Play Game Card */}
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle>Play</CardTitle>
              <CardDescription>
                Start a new four-in-a-row game
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/play">
                <Button className="w-full" size="lg">
                  New Game
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* My Games Card */}
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle>My Games</CardTitle>
              <CardDescription>
                View your game history and replays
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/games">
                <Button variant="outline" className="w-full" size="lg">
                  View History
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* Leaderboard Card */}
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle>Leaderboard</CardTitle>
              <CardDescription>
                See top players and rankings
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/leaderboard">
                <Button variant="outline" className="w-full" size="lg">
                  View Rankings
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* Statistics Card */}
          <Card className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle>Statistics</CardTitle>
              <CardDescription>
                View detailed analytics and trends
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/stats">
                <Button variant="outline" className="w-full" size="lg">
                  View Stats
                </Button>
              </Link>
            </CardContent>
          </Card>

          {/* Watch Live Games Card */}
          <Card className="hover:shadow-lg transition-shadow border-purple-500/30 bg-gradient-to-br from-purple-50/50 to-white dark:from-purple-950/30 dark:to-gray-900">
            <CardHeader>
              <div className="flex items-center gap-2">
                <CardTitle>Watch Games</CardTitle>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200">
                  Live
                </span>
              </div>
              <CardDescription>
                Watch live bot battles and player matches
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Link to="/spectate">
                <Button variant="outline" className="w-full border-purple-500/50 hover:bg-purple-50 dark:hover:bg-purple-950/50" size="lg">
                  Spectate Games
                </Button>
              </Link>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
