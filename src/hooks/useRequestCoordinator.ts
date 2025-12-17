/**
 * useRequestCoordinator - Coordinates async requests with polling
 *
 * This hook prevents race conditions between:
 * - Optimistic updates from user actions
 * - Polling responses that may contain stale data
 * - API responses that confirm/reject optimistic updates
 *
 * Key features:
 * - Tracks submission state to pause polling
 * - Provides shouldPoll() check for conditional polling
 * - Wraps async operations to manage submission lifecycle
 */

import { useState, useCallback, useRef, useEffect } from 'react'

export interface UseRequestCoordinatorReturn {
  /** Whether a submission is currently in progress */
  isSubmitting: boolean
  /** Whether polling should proceed (false during submission) */
  shouldPoll: () => boolean
  /** Wrap an async submission to track its lifecycle */
  withSubmission: <T>(fn: () => Promise<T>) => Promise<T>
  /** Manually pause polling (increments pause count) */
  pausePolling: () => void
  /** Manually resume polling (decrements pause count) */
  resumePolling: () => void
  /** Get current pause count (for debugging) */
  getPauseCount: () => number
}

export function useRequestCoordinator(): UseRequestCoordinatorReturn {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const pauseCountRef = useRef(0)
  const isSubmittingRef = useRef(false)

  // Keep ref in sync with state for shouldPoll to read synchronously
  useEffect(() => {
    isSubmittingRef.current = isSubmitting
  }, [isSubmitting])

  // Check if polling should proceed
  const shouldPoll = useCallback(() => {
    return !isSubmittingRef.current && pauseCountRef.current === 0
  }, [])

  // Wrap an async operation to track submission state
  const withSubmission = useCallback(async <T>(fn: () => Promise<T>): Promise<T> => {
    setIsSubmitting(true)
    isSubmittingRef.current = true

    try {
      return await fn()
    } finally {
      setIsSubmitting(false)
      isSubmittingRef.current = false
    }
  }, [])

  // Manually pause polling
  const pausePolling = useCallback(() => {
    pauseCountRef.current += 1
  }, [])

  // Manually resume polling
  const resumePolling = useCallback(() => {
    pauseCountRef.current = Math.max(0, pauseCountRef.current - 1)
  }, [])

  // Get pause count for debugging
  const getPauseCount = useCallback(() => {
    return pauseCountRef.current
  }, [])

  return {
    isSubmitting,
    shouldPoll,
    withSubmission,
    pausePolling,
    resumePolling,
    getPauseCount,
  }
}

/**
 * usePollingWithCoordination - Combines polling interval with request coordination
 *
 * This is a convenience hook that sets up an interval that respects
 * the coordinator's shouldPoll() check.
 */
export interface UsePollingConfig {
  /** Polling interval in milliseconds */
  interval: number
  /** The polling function to call */
  pollFn: () => Promise<void>
  /** Whether polling is enabled */
  enabled: boolean
  /** Request coordinator to check before polling */
  coordinator: UseRequestCoordinatorReturn
}

export function usePollingWithCoordination({
  interval,
  pollFn,
  enabled,
  coordinator,
}: UsePollingConfig) {
  const pollFnRef = useRef(pollFn)

  // Keep pollFn ref updated
  useEffect(() => {
    pollFnRef.current = pollFn
  }, [pollFn])

  useEffect(() => {
    if (!enabled) return

    const poll = async () => {
      // Skip if coordinator says not to poll
      if (!coordinator.shouldPoll()) return

      try {
        await pollFnRef.current()
      } catch (error) {
        console.error('Polling error:', error)
      }
    }

    // Initial poll
    poll()

    // Set up interval
    const intervalId = setInterval(poll, interval)

    return () => {
      clearInterval(intervalId)
    }
  }, [enabled, interval, coordinator])
}
