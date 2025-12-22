/**
 * Generic polling hook for interval-based data fetching
 *
 * Abstracts the common pattern of polling an API endpoint at regular intervals
 * with proper cleanup, mount tracking, and start/stop controls.
 */

import { useCallback, useEffect, useRef } from 'react'

interface UsePollingOptions {
  /**
   * Polling interval in milliseconds.
   */
  interval: number

  /**
   * Whether polling is enabled. When false, polling stops.
   * Defaults to true.
   */
  enabled?: boolean

  /**
   * Whether to fetch immediately when polling starts.
   * Defaults to true.
   */
  immediate?: boolean

  /**
   * Dependencies that should restart polling when changed.
   * The fetch function is automatically included.
   */
  deps?: unknown[]
}

interface UsePollingResult {
  /**
   * Manually start polling. Usually not needed as polling starts automatically.
   */
  start: () => void

  /**
   * Manually stop polling.
   */
  stop: () => void

  /**
   * Whether polling is currently active.
   */
  isPolling: boolean
}

/**
 * Hook for polling data at regular intervals with automatic cleanup.
 *
 * @param fetchFn - Async function to call on each poll
 * @param options - Polling configuration options
 * @returns Controls for starting/stopping polling
 *
 * @example
 * // Basic usage
 * const { isPolling } = usePolling(fetchMessages, {
 *   interval: CHAT_POLL_INTERVAL,
 *   enabled: isActive && !isMuted,
 * })
 *
 * @example
 * // With manual control
 * const { start, stop } = usePolling(fetchGameState, {
 *   interval: 1000,
 *   immediate: false,
 *   enabled: false, // Start manually
 * })
 */
export function usePolling(
  fetchFn: () => Promise<void> | void,
  options: UsePollingOptions
): UsePollingResult {
  const {
    interval,
    enabled = true,
    immediate = true,
    deps = [],
  } = options

  const pollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)
  const isPollingRef = useRef(false)

  // Stop polling helper
  const stop = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    isPollingRef.current = false
  }, [])

  // Start polling helper
  const start = useCallback(() => {
    // Clear any existing interval
    stop()

    if (!isMountedRef.current) return

    // Initial fetch if immediate is true
    if (immediate) {
      fetchFn()
    }

    // Start interval
    pollRef.current = setInterval(() => {
      if (isMountedRef.current) {
        fetchFn()
      }
    }, interval)

    isPollingRef.current = true
  }, [fetchFn, interval, immediate, stop])

  // Track mount state and cleanup
  useEffect(() => {
    isMountedRef.current = true

    return () => {
      isMountedRef.current = false
      stop()
    }
  }, [stop])

  // Auto start/stop based on enabled flag
  useEffect(() => {
    if (enabled) {
      start()
    } else {
      stop()
    }

    return () => stop()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, start, stop, ...deps])

  return {
    start,
    stop,
    isPolling: isPollingRef.current,
  }
}

/**
 * Hook that provides a ref to track if component is mounted.
 * Useful for preventing state updates after unmount in async operations.
 *
 * @example
 * const isMounted = useIsMounted()
 *
 * const fetchData = async () => {
 *   const data = await api.fetch()
 *   if (isMounted.current) {
 *     setData(data)
 *   }
 * }
 */
export function useIsMounted(): React.MutableRefObject<boolean> {
  const isMountedRef = useRef(true)

  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  return isMountedRef
}
