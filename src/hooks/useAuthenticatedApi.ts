/**
 * Custom React hook for making authenticated API calls
 *
 * Provides consistent session token handling and error management
 * across all components that need to interact with protected API endpoints.
 */

import { useCallback } from 'react'

interface ApiCallOptions extends RequestInit {
  headers?: HeadersInit
}

export function useAuthenticatedApi() {
  /**
   * Get the session token from localStorage
   */
  const getSessionToken = useCallback(() => {
    return localStorage.getItem('makefour_session_token')
  }, [])

  /**
   * Make an authenticated API call with automatic session token injection
   *
   * @param url - The API endpoint URL
   * @param options - Standard fetch RequestInit options
   * @returns Parsed JSON response
   * @throws Error if not authenticated or request fails
   */
  const apiCall = useCallback(async <T = any>(
    url: string,
    options: ApiCallOptions = {}
  ): Promise<T> => {
    const sessionToken = getSessionToken()

    if (!sessionToken) {
      throw new Error('Not authenticated')
    }

    const response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${sessionToken}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.details || errorData.error || 'Request failed')
    }

    return response.json()
  }, [getSessionToken])

  return {
    getSessionToken,
    apiCall,
  }
}
