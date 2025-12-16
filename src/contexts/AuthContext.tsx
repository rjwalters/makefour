import type React from 'react'
import { createContext, useContext, useState, useEffect } from 'react'
import type { PublicUser } from '../lib/schemas/auth'
import { decryptDEK } from '../lib/crypto'

interface AuthContextType {
  isAuthenticated: boolean
  user: PublicUser | null
  login: (email: string, password: string) => Promise<{ success: boolean; error?: string }>
  register: (email: string, password: string, username: string) => Promise<{ success: boolean; error?: string }>
  loginWithGoogle: () => void
  logout: () => Promise<void>
  resendVerification: () => Promise<{ success: boolean; error?: string }>
  refreshUser: () => Promise<void>
  encryptionKey: CryptoKey | null
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Helper to export CryptoKey to base64 for localStorage
async function exportKeyToStorage(key: CryptoKey): Promise<string> {
  const exported = await crypto.subtle.exportKey('raw', key)
  const exportedArray = Array.from(new Uint8Array(exported))
  return btoa(String.fromCharCode(...exportedArray))
}

// Helper to import CryptoKey from base64 localStorage
async function importKeyFromStorage(keyString: string): Promise<CryptoKey> {
  const keyData = Uint8Array.from(atob(keyString), c => c.charCodeAt(0))
  return crypto.subtle.importKey(
    'raw',
    keyData,
    { name: 'AES-GCM', length: 256 },
    true,
    ['encrypt', 'decrypt']
  )
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [user, setUser] = useState<PublicUser | null>(null)
  const [encryptionKey, setEncryptionKey] = useState<CryptoKey | null>(null)

  useEffect(() => {
    // Check for OAuth callback token in URL
    const handleOAuthCallback = async () => {
      const params = new URLSearchParams(window.location.search)
      const oauthToken = params.get('oauth_token')
      const oauthProvider = params.get('oauth_provider')

      if (oauthToken && oauthProvider) {
        // Clear the URL params
        window.history.replaceState({}, '', window.location.pathname)

        // Store the session token and fetch user data
        localStorage.setItem('makefour_session_token', oauthToken)

        try {
          const response = await fetch('/api/auth/me', {
            headers: { 'Authorization': `Bearer ${oauthToken}` }
          })

          if (response.ok) {
            const data = await response.json()
            setUser(data.user)
            setIsAuthenticated(true)
            console.log(`✓ OAuth login successful (${oauthProvider}):`, data.user.email)
          } else {
            localStorage.removeItem('makefour_session_token')
          }
        } catch (error) {
          console.error('OAuth callback failed:', error)
          localStorage.removeItem('makefour_session_token')
        }
        return
      }
    }

    handleOAuthCallback()
  }, [])

  useEffect(() => {
    // Check if user is already authenticated via session token
    const checkSession = async () => {
      const sessionToken = localStorage.getItem('makefour_session_token')
      const storedDEK = localStorage.getItem('makefour_dek')

      if (sessionToken) {
        try {
          const response = await fetch('/api/auth/me', {
            headers: {
              'Authorization': `Bearer ${sessionToken}`
            }
          })

          if (response.ok) {
            const data = await response.json()
            setUser(data.user)
            setIsAuthenticated(true)
            console.log('✓ Session restored for user:', data.user.email)

            // Restore encryption key from localStorage if available
            if (storedDEK) {
              try {
                const dek = await importKeyFromStorage(storedDEK)
                setEncryptionKey(dek)
                console.log('✓ Encryption key restored')
              } catch (error) {
                console.error('Failed to restore encryption key:', error)
                localStorage.removeItem('makefour_dek')
              }
            }
          } else {
            // Invalid/expired session, clear it
            localStorage.removeItem('makefour_session_token')
            localStorage.removeItem('makefour_dek')
          }
        } catch (error) {
          console.error('Session check failed:', error)
          localStorage.removeItem('makefour_session_token')
          localStorage.removeItem('makefour_dek')
        }
      }
    }

    checkSession()
  }, [])

  const register = async (email: string, password: string, username: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, username })
      })

      const data = await response.json()

      if (!response.ok) {
        return {
          success: false,
          error: data.details || data.error || 'Registration failed'
        }
      }

      console.log('✓ Registration successful:', data.user.email)

      // Now login with the same credentials
      return await login(email, password)
    } catch (error) {
      console.error('Registration failed:', error)
      return {
        success: false,
        error: 'An unexpected error occurred during registration'
      }
    }
  }

  const login = async (email: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })

      const data = await response.json()

      if (!response.ok) {
        return {
          success: false,
          error: data.details || data.error || 'Login failed'
        }
      }

      // Store session token
      localStorage.setItem('makefour_session_token', data.session_token)

      // Set user data
      setUser(data.user)
      setIsAuthenticated(true)

      console.log('✓ Login successful:', data.user.email)

      // Decrypt and store DEK (Data Encryption Key)
      if (data.encrypted_dek) {
        try {
          const dek = await decryptDEK(data.encrypted_dek, password)
          setEncryptionKey(dek)

          // Store DEK in localStorage for session persistence
          const exportedDEK = await exportKeyToStorage(dek)
          localStorage.setItem('makefour_dek', exportedDEK)

          console.log('✓ Encryption key decrypted and stored')
        } catch (error) {
          console.error('Failed to decrypt DEK:', error)
          // Continue anyway - auth is successful even if encryption setup fails
        }
      }

      return { success: true }
    } catch (error) {
      console.error('Login failed:', error)
      return {
        success: false,
        error: 'An unexpected error occurred during login'
      }
    }
  }

  const logout = async () => {
    const sessionToken = localStorage.getItem('makefour_session_token')

    if (sessionToken) {
      try {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${sessionToken}`
          }
        })
        console.log('✓ Logout successful')
      } catch (error) {
        console.error('Logout API call failed:', error)
      }
    }

    // Clear local state and stored keys
    localStorage.removeItem('makefour_session_token')
    localStorage.removeItem('makefour_dek')
    setUser(null)
    setEncryptionKey(null)
    setIsAuthenticated(false)
  }

  const loginWithGoogle = () => {
    // Redirect to Google OAuth endpoint
    window.location.href = '/api/auth/google'
  }

  const resendVerification = async (): Promise<{ success: boolean; error?: string }> => {
    const sessionToken = localStorage.getItem('makefour_session_token')

    if (!sessionToken) {
      return { success: false, error: 'Not authenticated' }
    }

    try {
      const response = await fetch('/api/auth/resend-verification', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${sessionToken}`,
        },
      })

      const data = await response.json()

      if (!response.ok) {
        return { success: false, error: data.error || 'Failed to resend verification email' }
      }

      return { success: true }
    } catch (error) {
      console.error('Resend verification failed:', error)
      return { success: false, error: 'An unexpected error occurred' }
    }
  }

  const refreshUser = async (): Promise<void> => {
    const sessionToken = localStorage.getItem('makefour_session_token')

    if (!sessionToken) {
      return
    }

    try {
      const response = await fetch('/api/auth/me', {
        headers: { 'Authorization': `Bearer ${sessionToken}` },
      })

      if (response.ok) {
        const data = await response.json()
        setUser(data.user)
      }
    } catch (error) {
      console.error('Failed to refresh user:', error)
    }
  }

  return (
    <AuthContext.Provider
      value={{
        isAuthenticated,
        user,
        login,
        register,
        loginWithGoogle,
        logout,
        resendVerification,
        refreshUser,
        encryptionKey,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
