import { Link } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Button } from '../components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import ThemeToggle from '../components/ThemeToggle'

export default function DashboardPage() {
  const { logout, user } = useAuth()

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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
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
