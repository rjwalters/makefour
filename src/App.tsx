import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from './contexts/ThemeContext'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { PreferencesProvider } from './contexts/PreferencesContext'
import { ToastProvider } from './contexts/ToastContext'
import ToastContainer from './components/Toast'
import Footer from './components/Footer'
import LoginPage from './pages/LoginPage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import ResetPasswordPage from './pages/ResetPasswordPage'
import PlayPage from './pages/PlayPage'
import GamesPage from './pages/GamesPage'
import ReplayPage from './pages/ReplayPage'
import LeaderboardPage from './pages/LeaderboardPage'
import SpectatorPage from './pages/SpectatorPage'
import ProfilePage from './pages/ProfilePage'
import StatsPage from './pages/StatsPage'
import VerifyEmailPage from './pages/VerifyEmailPage'
import BotProfilePage from './pages/BotProfilePage'

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth()
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />
}

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <PreferencesProvider>
          <ToastProvider>
            <BrowserRouter>
              <Routes>
            {/* Public routes */}
            <Route path="/" element={<PlayPage />} />
            <Route path="/play" element={<PlayPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/forgot-password" element={<ForgotPasswordPage />} />
            <Route path="/reset-password" element={<ResetPasswordPage />} />
            <Route path="/spectate" element={<SpectatorPage />} />
            <Route path="/verify-email" element={<VerifyEmailPage />} />
            <Route path="/bot/:id" element={<BotProfilePage />} />

            {/* Protected routes - require authentication */}
            {/* Redirect old dashboard URL to profile */}
            <Route path="/dashboard" element={<Navigate to="/profile" replace />} />
            <Route
              path="/games"
              element={
                <ProtectedRoute>
                  <GamesPage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/replay/:gameId"
              element={
                <ProtectedRoute>
                  <ReplayPage />
                </ProtectedRoute>
              }
            />
            <Route path="/leaderboard" element={<LeaderboardPage />} />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <ProfilePage />
                </ProtectedRoute>
              }
            />
            <Route
              path="/stats"
              element={
                <ProtectedRoute>
                  <StatsPage />
                </ProtectedRoute>
              }
            />
              </Routes>
              <Footer />
              <ToastContainer />
            </BrowserRouter>
          </ToastProvider>
        </PreferencesProvider>
      </AuthProvider>
    </ThemeProvider>
  )
}

export default App
