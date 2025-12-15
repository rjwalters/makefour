/**
 * Custom React hook for spectating game chat
 *
 * Similar to useGameChat but for spectators - read-only, no authentication required.
 * Works for bot vs bot games and other spectatable matches.
 */

import { useState, useCallback, useEffect, useRef } from 'react'

export interface SpectatorChatMessage {
  id: string
  senderName: string
  senderType: 'human' | 'bot'
  content: string
  createdAt: number
}

interface SpectatorChatState {
  messages: SpectatorChatMessage[]
  isLoading: boolean
  error: string | null
}

const CHAT_POLL_INTERVAL = 1000 // 1 second for responsive chat

export function useSpectatorChat(gameId: string | null, isActive: boolean = true) {
  const [state, setState] = useState<SpectatorChatState>({
    messages: [],
    isLoading: false,
    error: null,
  })

  const pollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)
  const lastMessageTimeRef = useRef(0)

  // Cleanup on unmount
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
      if (pollRef.current) {
        clearInterval(pollRef.current)
      }
    }
  }, [])

  /**
   * Fetch messages from the server (no auth required)
   */
  const fetchMessages = useCallback(async () => {
    if (!gameId) return

    try {
      const since = lastMessageTimeRef.current
      const response = await fetch(`/api/games/${gameId}/chat?since=${since}`)

      if (!response.ok) {
        if (response.status === 403) {
          // Game not spectatable
          return
        }
        throw new Error('Failed to fetch chat')
      }

      const data = await response.json() as {
        messages: SpectatorChatMessage[]
        gameStatus: string
      }

      if (!isMountedRef.current) return

      if (data.messages.length > 0) {
        // Update last message time
        const latestTime = Math.max(...data.messages.map(m => m.createdAt))
        lastMessageTimeRef.current = latestTime

        setState(prev => {
          // Merge new messages, avoiding duplicates
          const existingIds = new Set(prev.messages.map(m => m.id))
          const newMessages = data.messages.filter(m => !existingIds.has(m.id))

          return {
            ...prev,
            messages: [...prev.messages, ...newMessages],
            error: null,
          }
        })
      }
    } catch (error) {
      // Don't set error state for transient polling failures
      console.error('Spectator chat poll error:', error)
    }
  }, [gameId])

  /**
   * Start polling for messages
   */
  const startPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
    }

    // Initial fetch
    fetchMessages()

    // Poll on interval
    pollRef.current = setInterval(fetchMessages, CHAT_POLL_INTERVAL)
  }, [fetchMessages])

  /**
   * Stop polling for messages
   */
  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  // Start/stop polling based on gameId and active state
  useEffect(() => {
    if (gameId && isActive) {
      startPolling()
    } else {
      stopPolling()
    }

    return () => stopPolling()
  }, [gameId, isActive, startPolling, stopPolling])

  /**
   * Clear messages (for switching games)
   */
  const clearMessages = useCallback(() => {
    lastMessageTimeRef.current = 0
    setState({
      messages: [],
      isLoading: false,
      error: null,
    })
  }, [])

  // Clear messages when game changes
  useEffect(() => {
    clearMessages()
  }, [gameId, clearMessages])

  return {
    ...state,
    clearMessages,
  }
}
