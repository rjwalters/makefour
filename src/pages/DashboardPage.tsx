import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'

export default function DashboardPage() {
  const { logout, user } = useAuth()

  // Calculate win rate
  const winRate = user && user.gamesPlayed > 0
    ? Math.round((user.wins / user.gamesPlayed) * 100)
    : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold">MakeFour</h1>
            {user && (
              <p className="text-xs text-muted-foreground">{user.email}</p>
            )}
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

          {/* AI Coach Card (Coming Soon) */}
          <Card className="hover:shadow-lg transition-shadow opacity-75">
            <CardHeader>
              <CardTitle>AI Coach</CardTitle>
              <CardDescription>
                Get move suggestions and game analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button variant="secondary" className="w-full" size="lg" disabled>
                Coming Soon
              </Button>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
