import type React from 'react'
import { createContext, useContext, useState, useEffect } from 'react'
import type { PublicUser } from '../lib/schemas/auth'
import { decryptDEK } from '../lib/crypto'
import { STORAGE_KEY_SESSION_TOKEN, STORAGE_KEY_DEK } from '../lib/storageKeys'
import { API_AUTH } from '../lib/apiEndpoints'

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
        localStorage.setItem(STORAGE_KEY_SESSION_TOKEN, oauthToken)

        try {
          const response = await fetch(API_AUTH.ME, {
            headers: { 'Authorization': `Bearer ${oauthToken}` }
          })

          if (response.ok) {
            const data = await response.json()
            setUser(data.user)
            setIsAuthenticated(true)
            console.log(`✓ OAuth login successful (${oauthProvider}):`, data.user.email)
          } else {
            localStorage.removeItem(STORAGE_KEY_SESSION_TOKEN)
          }
        } catch (error) {
          console.error('OAuth callback failed:', error)
          localStorage.removeItem(STORAGE_KEY_SESSION_TOKEN)
        }
        return
      }
    }

    handleOAuthCallback()
  }, [])

  useEffect(() => {
    // Check if user is already authenticated via session token
    const checkSession = async () => {
      const sessionToken = localStorage.getItem(STORAGE_KEY_SESSION_TOKEN)
      const storedDEK = localStorage.getItem(STORAGE_KEY_DEK)

      if (sessionToken) {
        try {
          const response = await fetch(API_AUTH.ME, {
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
                localStorage.removeItem(STORAGE_KEY_DEK)
              }
            }
          } else {
            // Invalid/expired session, clear it
            localStorage.removeItem(STORAGE_KEY_SESSION_TOKEN)
            localStorage.removeItem(STORAGE_KEY_DEK)
          }
        } catch (error) {
          console.error('Session check failed:', error)
          localStorage.removeItem(STORAGE_KEY_SESSION_TOKEN)
          localStorage.removeItem(STORAGE_KEY_DEK)
        }
      }
    }

    checkSession()
  }, [])

  const register = async (email: string, password: string, username: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const response = await fetch(API_AUTH.REGISTER, {
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
      const response = await fetch(API_AUTH.LOGIN, {
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
      localStorage.setItem(STORAGE_KEY_SESSION_TOKEN, data.session_token)

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
          localStorage.setItem(STORAGE_KEY_DEK, exportedDEK)

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
    const sessionToken = localStorage.getItem(STORAGE_KEY_SESSION_TOKEN)

    if (sessionToken) {
      try {
        await fetch(API_AUTH.LOGOUT, {
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
    localStorage.removeItem(STORAGE_KEY_SESSION_TOKEN)
    localStorage.removeItem(STORAGE_KEY_DEK)
    setUser(null)
    setEncryptionKey(null)
    setIsAuthenticated(false)
  }

  const loginWithGoogle = () => {
    // Redirect to Google OAuth endpoint
    window.location.href = API_AUTH.GOOGLE
  }

  const resendVerification = async (): Promise<{ success: boolean; error?: string }> => {
    const sessionToken = localStorage.getItem(STORAGE_KEY_SESSION_TOKEN)

    if (!sessionToken) {
      return { success: false, error: 'Not authenticated' }
    }

    try {
      const response = await fetch(API_AUTH.RESEND_VERIFICATION, {
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
    const sessionToken = localStorage.getItem(STORAGE_KEY_SESSION_TOKEN)

    if (!sessionToken) {
      return
    }

    try {
      const response = await fetch(API_AUTH.ME, {
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
