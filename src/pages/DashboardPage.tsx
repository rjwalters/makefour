import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'
import { VerificationBanner } from '../components/VerificationBanner'

export default function DashboardPage() {
  const { logout, user } = useAuth()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  // Calculate win rate
  const winRate = user && user.gamesPlayed > 0
    ? Math.round((user.wins / user.gamesPlayed) * 100)
    : 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3 sm:py-4 flex justify-between items-center">
          <div className="min-w-0 flex-shrink">
            <Link to="/dashboard" className="text-xl sm:text-2xl font-bold hover:opacity-80">
              MakeFour
            </Link>
            {user && (
              <p className="text-xs text-muted-foreground truncate max-w-[150px] sm:max-w-none">{user.email}</p>
            )}
          </div>

          {/* Desktop navigation */}
          <div className="hidden sm:flex gap-2">
            <Link to="/stats">
              <Button variant="ghost" size="sm">Stats</Button>
            </Link>
            <Link to="/profile">
              <Button variant="ghost" size="sm">Profile</Button>
            </Link>
            <ThemeToggle />
            <Button variant="outline" onClick={logout} size="sm">
              Logout
            </Button>
          </div>

          {/* Mobile navigation */}
          <div className="flex sm:hidden gap-2 items-center">
            <ThemeToggle />
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 touch-manipulation"
              aria-label="Toggle menu"
              aria-expanded={mobileMenuOpen}
            >
              {mobileMenuOpen ? (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              ) : (
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu dropdown */}
        {mobileMenuOpen && (
          <div className="sm:hidden border-t bg-white dark:bg-gray-800 px-4 py-3 space-y-2">
            <Link
              to="/play"
              className="block"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Button className="w-full justify-start h-12 touch-manipulation">
                Play
              </Button>
            </Link>
            <Link
              to="/games"
              className="block"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Button variant="outline" className="w-full justify-start h-12 touch-manipulation">
                My Games
              </Button>
            </Link>
            <Link
              to="/leaderboard"
              className="block"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Button variant="outline" className="w-full justify-start h-12 touch-manipulation">
                Leaderboard
              </Button>
            </Link>
            <Link
              to="/stats"
              className="block"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Button variant="outline" className="w-full justify-start h-12 touch-manipulation">
                Stats
              </Button>
            </Link>
            <Link
              to="/profile"
              className="block"
              onClick={() => setMobileMenuOpen(false)}
            >
              <Button variant="outline" className="w-full justify-start h-12 touch-manipulation">
                Profile
              </Button>
            </Link>
            <Button
              variant="outline"
              onClick={() => { logout(); setMobileMenuOpen(false); }}
              className="w-full justify-start h-12 touch-manipulation"
            >
              Logout
            </Button>
          </div>
        )}
      </header>

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
        </div>
      </main>
    </div>
  )
}
