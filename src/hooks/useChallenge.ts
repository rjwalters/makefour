/**
 * Custom React hook for managing player challenges
 *
 * Handles:
 * - Creating challenges to specific players
 * - Polling for incoming challenges
 * - Accepting/declining challenges
 * - Showing toast notifications for incoming challenges
 */

import { useState, useCallback, useRef } from 'react'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import { usePolling, useIsMounted } from './usePolling'
import { useToast } from '../contexts/ToastContext'
import { useAuth } from '../contexts/AuthContext'
import { CHALLENGE_POLL_INTERVAL } from '../lib/pollingConstants'

export type ChallengeStatus = 'idle' | 'sending' | 'waiting' | 'matched' | 'error'

export interface OutgoingChallenge {
  id: string
  targetUsername: string
  targetRating: number | null
  targetExists: boolean
  status: 'pending' | 'accepted' | 'cancelled' | 'declined' | 'expired'
  createdAt: number
  expiresAt: number
  gameId: string | null
}

export interface IncomingChallenge {
  id: string
  challengerUsername: string
  challengerRating: number
  createdAt: number
  expiresAt: number
}

export interface MatchedGame {
  gameId: string
  opponentUsername: string
  opponentRating: number | null
  playerNumber: 1 | 2
}

interface ChallengeState {
  status: ChallengeStatus
  error: string | null
  outgoingChallenge: OutgoingChallenge | null
  incomingChallenges: IncomingChallenge[]
  matchedGame: MatchedGame | null
}

export function useChallenge() {
  const { apiCall } = useAuthenticatedApi()
  const { showToast, dismissToast } = useToast()
  const { isAuthenticated } = useAuth()
  const isMounted = useIsMounted()

  const [state, setState] = useState<ChallengeState>({
    status: 'idle',
    error: null,
    outgoingChallenge: null,
    incomingChallenges: [],
    matchedGame: null,
  })

  const shownChallengeIds = useRef<Set<string>>(new Set())
  const toastIds = useRef<Map<string, string>>(new Map())

  /**
   * Poll for incoming challenges
   */
  const pollIncoming = useCallback(async () => {
    if (!isAuthenticated) return

    try {
      const response = await apiCall<{
        incoming: IncomingChallenge[]
        matchedGame: { gameId: string; opponentUsername: string; opponentRating: number } | null
      }>('/api/challenges/incoming')

      if (!isMounted.current) return

      // Check for new challenges and show toasts
      for (const challenge of response.incoming) {
        if (!shownChallengeIds.current.has(challenge.id)) {
          shownChallengeIds.current.add(challenge.id)

          const toastId = showToast({
            type: 'challenge',
            title: `${challenge.challengerUsername} wants to play!`,
            message: `Rating: ${challenge.challengerRating}`,
            duration: 0, // Persist until action
            action: {
              label: 'Accept',
              onClick: () => acceptChallenge(challenge.id),
            },
            secondaryAction: {
              label: 'Decline',
              onClick: () => declineChallenge(challenge.id),
            },
            data: { challengeId: challenge.id },
          })

          toastIds.current.set(challenge.id, toastId)
        }
      }

      // Remove toasts for challenges that no longer exist
      const currentIds = new Set(response.incoming.map((c) => c.id))
      for (const [challengeId, toastId] of toastIds.current.entries()) {
        if (!currentIds.has(challengeId)) {
          dismissToast(toastId)
          toastIds.current.delete(challengeId)
          shownChallengeIds.current.delete(challengeId)
        }
      }

      // Check if our outgoing challenge was matched
      if (response.matchedGame) {
        setState((prev) => ({
          ...prev,
          status: 'matched',
          matchedGame: {
            gameId: response.matchedGame!.gameId,
            opponentUsername: response.matchedGame!.opponentUsername,
            opponentRating: response.matchedGame!.opponentRating,
            playerNumber: 1, // Challenger is always player 1
          },
          outgoingChallenge: null,
        }))
        return // Stop processing, polling will be disabled by state change
      }

      setState((prev) => ({
        ...prev,
        incomingChallenges: response.incoming,
      }))
    } catch (error) {
      // Silently ignore polling errors
      console.error('Challenge polling error:', error)
    }
  }, [apiCall, isAuthenticated, showToast, dismissToast, isMounted])

  // Poll for incoming challenges when authenticated and not matched
  const shouldPoll = isAuthenticated && state.status !== 'matched'
  usePolling(pollIncoming, {
    interval: CHALLENGE_POLL_INTERVAL,
    enabled: shouldPoll,
  })

  /**
   * Send a challenge to a specific player
   */
  const sendChallenge = useCallback(
    async (targetUsername: string) => {
      setState((prev) => ({ ...prev, status: 'sending', error: null }))

      try {
        const response = await apiCall<{
          status: 'pending' | 'matched'
          challengeId?: string
          gameId?: string
          target?: { username: string; rating: number | null; exists: boolean }
          expiresAt?: number
          opponent?: { id: string; username: string; rating: number }
        }>('/api/challenges', {
          method: 'POST',
          body: JSON.stringify({ targetUsername }),
        })

        if (!isMounted.current) return

        if (response.status === 'matched') {
          // Mutual challenge - game started immediately
          setState((prev) => ({
            ...prev,
            status: 'matched',
            matchedGame: {
              gameId: response.gameId!,
              opponentUsername: response.opponent!.username,
              opponentRating: response.opponent!.rating,
              playerNumber: 1, // Challenger is player 1
            },
          }))
        } else {
          // Challenge sent, waiting for acceptance
          setState((prev) => ({
            ...prev,
            status: 'waiting',
            outgoingChallenge: {
              id: response.challengeId!,
              targetUsername: response.target!.username,
              targetRating: response.target!.rating,
              targetExists: response.target!.exists,
              status: 'pending',
              createdAt: Date.now(),
              expiresAt: response.expiresAt!,
              gameId: null,
            },
          }))
        }
      } catch (error) {
        if (!isMounted.current) return
        setState((prev) => ({
          ...prev,
          status: 'error',
          error: error instanceof Error ? error.message : 'Failed to send challenge',
        }))
      }
    },
    [apiCall]
  )

  /**
   * Cancel outgoing challenge
   */
  const cancelChallenge = useCallback(async () => {
    const challengeId = state.outgoingChallenge?.id
    if (!challengeId) return

    try {
      await apiCall(`/api/challenges/${challengeId}`, { method: 'DELETE' })
    } catch {
      // Ignore errors
    }

    if (!isMounted.current) return

    setState((prev) => ({
      ...prev,
      status: 'idle',
      outgoingChallenge: null,
      error: null,
    }))
  }, [apiCall, state.outgoingChallenge])

  /**
   * Accept an incoming challenge
   */
  const acceptChallenge = useCallback(
    async (challengeId: string) => {
      // Dismiss the toast
      const toastId = toastIds.current.get(challengeId)
      if (toastId) {
        dismissToast(toastId)
        toastIds.current.delete(challengeId)
      }

      try {
        const response = await apiCall<{
          status: string
          gameId: string
          playerNumber: 1 | 2
          opponent: { username: string; rating: number }
        }>(`/api/challenges/${challengeId}/accept`, { method: 'POST' })

        if (!isMounted.current) return

        setState((prev) => ({
          ...prev,
          status: 'matched',
          matchedGame: {
            gameId: response.gameId,
            opponentUsername: response.opponent.username,
            opponentRating: response.opponent.rating,
            playerNumber: response.playerNumber,
          },
          incomingChallenges: prev.incomingChallenges.filter((c) => c.id !== challengeId),
        }))
      } catch (error) {
        if (!isMounted.current) return
        setState((prev) => ({
          ...prev,
          error: error instanceof Error ? error.message : 'Failed to accept challenge',
        }))
      }
    },
    [apiCall, dismissToast]
  )

  /**
   * Decline an incoming challenge
   */
  const declineChallenge = useCallback(
    async (challengeId: string) => {
      // Dismiss the toast
      const toastId = toastIds.current.get(challengeId)
      if (toastId) {
        dismissToast(toastId)
        toastIds.current.delete(challengeId)
      }
      shownChallengeIds.current.delete(challengeId)

      try {
        await apiCall(`/api/challenges/${challengeId}`, { method: 'DELETE' })
      } catch {
        // Ignore errors
      }

      if (!isMounted.current) return

      setState((prev) => ({
        ...prev,
        incomingChallenges: prev.incomingChallenges.filter((c) => c.id !== challengeId),
      }))
    },
    [apiCall, dismissToast]
  )

  /**
   * Reset to idle state
   */
  const reset = useCallback(() => {
    setState({
      status: 'idle',
      error: null,
      outgoingChallenge: null,
      incomingChallenges: [],
      matchedGame: null,
    })
    // Polling will automatically restart via usePolling when status becomes 'idle'
  }, [])

  return {
    ...state,
    sendChallenge,
    cancelChallenge,
    acceptChallenge,
    declineChallenge,
    reset,
  }
}
