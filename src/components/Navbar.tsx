/**
 * Shared navigation bar component
 *
 * Displays consistent navigation across all pages with:
 * - Logo linking to home
 * - Nav links: Leaderboard, Watch, Coach, Compete (auth only)
 * - Theme and sound toggles
 * - User menu dropdown (auth) or Sign In button (anon)
 * - Mobile hamburger menu
 */

import { useState, useRef, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useSounds } from '../hooks/useSounds'
import ThemeToggle from './ThemeToggle'
import SoundToggle from './SoundToggle'
import { Button } from './ui/button'

export default function Navbar() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [userMenuOpen, setUserMenuOpen] = useState(false)
  const { isAuthenticated, user, logout } = useAuth()
  const location = useLocation()
  const sounds = useSounds()
  const userMenuRef = useRef<HTMLDivElement>(null)

  // Close user menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (userMenuRef.current && !userMenuRef.current.contains(event.target as Node)) {
        setUserMenuOpen(false)
      }
    }

    if (userMenuOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [userMenuOpen])

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  const isActive = (path: string) => location.pathname === path

  const navLinkClass = (path: string) =>
    `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive(path)
        ? 'bg-primary text-primary-foreground'
        : 'text-muted-foreground hover:text-foreground hover:bg-muted'
    }`

  const navLinks = [
    { path: '/leaderboard', label: 'Leaderboard' },
    { path: '/spectate', label: 'Watch' },
    { path: '/play?mode=training', label: 'Coach', matchPath: '/play' },
  ]

  const authNavLinks = [
    { path: '/play?mode=compete', label: 'Compete', matchPath: '/play' },
  ]

  const isNavActive = (link: { path: string; matchPath?: string }) => {
    if (link.matchPath) {
      return location.pathname === link.matchPath && location.search.includes(link.path.split('?')[1])
    }
    return isActive(link.path)
  }

  return (
    <header className="border-b bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3 sm:py-4">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <Link to="/" className="text-xl sm:text-2xl font-bold hover:opacity-80">
            MakeFour
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={navLinkClass(isNavActive(link) ? link.path : '')}
              >
                {link.label}
              </Link>
            ))}
            {isAuthenticated &&
              authNavLinks.map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={navLinkClass(isNavActive(link) ? link.path : '')}
                >
                  {link.label}
                </Link>
              ))}
          </nav>

          {/* Right side controls */}
          <div className="flex items-center gap-2">
            {/* Sound toggle - desktop only */}
            <div className="hidden sm:block">
              <SoundToggle
                settings={sounds.settings}
                onToggle={sounds.toggleSound}
                onVolumeChange={(volume) => sounds.updateSettings({ volume })}
              />
            </div>

            <ThemeToggle />

            {/* User menu or Sign In - desktop */}
            <div className="hidden md:block">
              {isAuthenticated ? (
                <div ref={userMenuRef} className="relative">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setUserMenuOpen(!userMenuOpen)}
                    className="gap-2"
                  >
                    <span className="max-w-[100px] truncate">
                      {user?.displayName || user?.email?.split('@')[0] || 'Account'}
                    </span>
                    <svg
                      className={`w-4 h-4 transition-transform ${userMenuOpen ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </Button>

                  {userMenuOpen && (
                    <div className="absolute right-0 top-full mt-2 w-48 bg-background border rounded-lg shadow-lg py-1 z-50">
                      <Link
                        to="/dashboard"
                        onClick={() => setUserMenuOpen(false)}
                        className="block px-4 py-2 text-sm hover:bg-muted"
                      >
                        Dashboard
                      </Link>
                      <Link
                        to="/profile"
                        onClick={() => setUserMenuOpen(false)}
                        className="block px-4 py-2 text-sm hover:bg-muted"
                      >
                        Profile
                      </Link>
                      <hr className="my-1 border-border" />
                      <button
                        onClick={() => {
                          logout()
                          setUserMenuOpen(false)
                        }}
                        className="block w-full text-left px-4 py-2 text-sm hover:bg-muted text-red-600 dark:text-red-400"
                      >
                        Logout
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                <Link to="/login">
                  <Button variant="outline" size="sm">
                    Sign In
                  </Button>
                </Link>
              )}
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 touch-manipulation"
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
      </div>

      {/* Mobile menu dropdown */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t bg-white dark:bg-gray-800 px-4 py-3 space-y-2">
          {/* Navigation links */}
          {navLinks.map((link) => (
            <Link key={link.path} to={link.path} className="block">
              <Button
                variant={isNavActive(link) ? 'default' : 'outline'}
                className="w-full justify-start h-12 touch-manipulation"
              >
                {link.label}
              </Button>
            </Link>
          ))}

          {isAuthenticated && (
            <>
              {authNavLinks.map((link) => (
                <Link key={link.path} to={link.path} className="block">
                  <Button
                    variant={isNavActive(link) ? 'default' : 'outline'}
                    className="w-full justify-start h-12 touch-manipulation"
                  >
                    {link.label}
                  </Button>
                </Link>
              ))}

              <hr className="my-2 border-border" />

              <Link to="/dashboard" className="block">
                <Button variant="outline" className="w-full justify-start h-12 touch-manipulation">
                  Dashboard
                </Button>
              </Link>
              <Link to="/profile" className="block">
                <Button variant="outline" className="w-full justify-start h-12 touch-manipulation">
                  Profile
                </Button>
              </Link>
              <Button
                variant="outline"
                onClick={logout}
                className="w-full justify-start h-12 touch-manipulation text-red-600 dark:text-red-400"
              >
                Logout
              </Button>
            </>
          )}

          {!isAuthenticated && (
            <Link to="/login" className="block">
              <Button variant="default" className="w-full justify-start h-12 touch-manipulation">
                Sign In
              </Button>
            </Link>
          )}

          {/* Sound toggle for mobile */}
          <div className="pt-2 sm:hidden">
            <SoundToggle
              settings={sounds.settings}
              onToggle={sounds.toggleSound}
              onVolumeChange={(volume) => sounds.updateSettings({ volume })}
            />
          </div>
        </div>
      )}
    </header>
  )
}
