/**
 * Generic data fetching hook with consistent loading, error, and refetch handling
 *
 * Reduces boilerplate for hooks that fetch data from API endpoints.
 * Supports both authenticated and unauthenticated requests.
 */

import { useState, useEffect, useCallback, useRef } from 'react'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import { getErrorMessage } from '../lib/errorUtils'

interface UseApiOptions<T, R = T> {
  /**
   * Whether to require authentication for this request.
   * If true, uses apiCall with auth headers. If false, uses plain fetch.
   */
  authenticated?: boolean

  /**
   * Whether to fetch data immediately on mount.
   * Defaults to true.
   */
  fetchOnMount?: boolean

  /**
   * Dependencies that should trigger a refetch when changed.
   * These are in addition to the URL.
   */
  deps?: unknown[]

  /**
   * Transform function to extract/transform data from the response.
   * Useful when the API response wraps data in an object.
   */
  transform?: (response: R) => T

  /**
   * Skip fetching if this condition is true.
   * Useful for conditional fetching based on auth state or other conditions.
   */
  skip?: boolean

  /**
   * Custom error message to use instead of the response error.
   */
  errorMessage?: string

  /**
   * Initial data value before first fetch.
   */
  initialData?: T
}

interface UseApiResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
}

/**
 * Generic hook for fetching data from an API endpoint
 *
 * @example
 * // Simple unauthenticated fetch
 * const { data, loading, error } = useApi<BotPersona[]>('/api/bot/personas', {
 *   transform: (res) => res.personas
 * })
 *
 * @example
 * // Authenticated fetch with dependencies
 * const { data, loading, error, refetch } = useApi<UserStats>(
 *   '/api/users/me/stats',
 *   { authenticated: true }
 * )
 *
 * @example
 * // Conditional fetch
 * const { data } = useApi<BotStats>(
 *   `/api/users/me/bot-stats/${botId}`,
 *   { authenticated: true, skip: !botId }
 * )
 */
export function useApi<T, R = T>(
  url: string | null,
  options: UseApiOptions<T, R> = {}
): UseApiResult<T> {
  const {
    authenticated = false,
    fetchOnMount = true,
    deps = [],
    transform,
    skip = false,
    errorMessage,
    initialData = null,
  } = options

  const { apiCall, getSessionToken } = useAuthenticatedApi()
  const [data, setData] = useState<T | null>(initialData)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Track if component is mounted to prevent state updates after unmount
  const isMountedRef = useRef(true)

  const fetchData = useCallback(async () => {
    // Skip if no URL, skip flag is true, or auth required but no token
    if (!url || skip) {
      return
    }

    if (authenticated && !getSessionToken()) {
      setData(null)
      return
    }

    setLoading(true)
    setError(null)

    try {
      let response: R

      if (authenticated) {
        response = await apiCall<R>(url)
      } else {
        const fetchResponse = await fetch(url)
        if (!fetchResponse.ok) {
          const errorData = await fetchResponse.json().catch(() => ({}))
          throw new Error(errorData.error || errorData.message || `Request failed with status ${fetchResponse.status}`)
        }
        response = await fetchResponse.json()
      }

      if (isMountedRef.current) {
        const transformedData = transform ? transform(response) : (response as unknown as T)
        setData(transformedData)
        setError(null)
      }
    } catch (err) {
      if (isMountedRef.current) {
        setError(errorMessage || getErrorMessage(err, 'Unknown error'))
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, authenticated, skip, apiCall, getSessionToken, transform, errorMessage, ...deps])

  useEffect(() => {
    isMountedRef.current = true

    if (fetchOnMount) {
      fetchData()
    }

    return () => {
      isMountedRef.current = false
    }
  }, [fetchData, fetchOnMount])

  return {
    data,
    loading,
    error,
    refetch: fetchData,
  }
}

/**
 * Convenience hook for authenticated API calls
 * Equivalent to useApi with authenticated: true
 */
export function useAuthenticatedApiCall<T, R = T>(
  url: string | null,
  options: Omit<UseApiOptions<T, R>, 'authenticated'> = {}
): UseApiResult<T> {
  return useApi<T, R>(url, { ...options, authenticated: true })
}
