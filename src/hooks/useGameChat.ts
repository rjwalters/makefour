/**
 * Custom React hook for managing in-game chat
 *
 * Handles:
 * - Polling for new messages
 * - Sending messages
 * - Bot response handling
 * - Mute/unmute functionality
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useAuthenticatedApi } from './useAuthenticatedApi'

export interface ChatMessage {
  id: string
  game_id: string
  sender_id: string
  sender_type: 'human' | 'bot'
  content: string
  created_at: number
}

interface ChatState {
  messages: ChatMessage[]
  isLoading: boolean
  isSending: boolean
  error: string | null
  isMuted: boolean
  unreadCount: number
}

const CHAT_POLL_INTERVAL = 1000 // 1 second for responsive chat

export function useGameChat(gameId: string | null, isActive: boolean = true) {
  const { apiCall } = useAuthenticatedApi()
  const [state, setState] = useState<ChatState>({
    messages: [],
    isLoading: false,
    isSending: false,
    error: null,
    isMuted: false,
    unreadCount: 0,
  })

  const pollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)
  const lastMessageTimeRef = useRef(0)
  const isVisibleRef = useRef(true) // Track if chat panel is visible

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
   * Fetch messages from the server
   */
  const fetchMessages = useCallback(async () => {
    if (!gameId || state.isMuted) return

    try {
      const since = lastMessageTimeRef.current
      const response = await apiCall<{
        messages: ChatMessage[]
        gameStatus: string
      }>(`/api/match/${gameId}/chat?since=${since}`)

      if (!isMountedRef.current) return

      if (response.messages.length > 0) {
        // Update last message time
        const latestTime = Math.max(...response.messages.map(m => m.created_at))
        lastMessageTimeRef.current = latestTime

        setState(prev => {
          // Merge new messages, avoiding duplicates
          const existingIds = new Set(prev.messages.map(m => m.id))
          const newMessages = response.messages.filter(m => !existingIds.has(m.id))

          // Calculate unread count if chat is not visible
          const unreadCount = isVisibleRef.current ? 0 : prev.unreadCount + newMessages.length

          return {
            ...prev,
            messages: [...prev.messages, ...newMessages],
            unreadCount,
            error: null,
          }
        })
      }
    } catch (error) {
      // Don't set error state for transient polling failures
      console.error('Chat poll error:', error)
    }
  }, [apiCall, gameId, state.isMuted])

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
    if (gameId && isActive && !state.isMuted) {
      startPolling()
    } else {
      stopPolling()
    }

    return () => stopPolling()
  }, [gameId, isActive, state.isMuted, startPolling, stopPolling])

  /**
   * Send a message
   */
  const sendMessage = useCallback(async (content: string) => {
    if (!gameId || !content.trim()) return false

    setState(prev => ({ ...prev, isSending: true, error: null }))

    try {
      const response = await apiCall<{
        success: boolean
        messageId: string
        botResponse: string | null
      }>(`/api/match/${gameId}/chat`, {
        method: 'POST',
        body: JSON.stringify({ content: content.trim() }),
      })

      if (!isMountedRef.current) return false

      // Add the user's message optimistically (will be deduplicated by poll)
      const userMessage: ChatMessage = {
        id: response.messageId,
        game_id: gameId,
        sender_id: 'self', // Will be replaced by actual ID from server
        sender_type: 'human',
        content: content.trim(),
        created_at: Date.now(),
      }

      setState(prev => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        isSending: false,
      }))

      // Trigger immediate poll to get any bot response
      setTimeout(fetchMessages, 100)

      return true
    } catch (error) {
      if (!isMountedRef.current) return false

      const errorMessage = error instanceof Error ? error.message : 'Failed to send message'

      // Check for rate limit
      if (errorMessage.includes('Rate limit')) {
        setState(prev => ({
          ...prev,
          isSending: false,
          error: 'Please wait before sending more messages',
        }))
      } else {
        setState(prev => ({
          ...prev,
          isSending: false,
          error: errorMessage,
        }))
      }

      return false
    }
  }, [apiCall, gameId, fetchMessages])

  /**
   * Toggle mute state
   */
  const toggleMute = useCallback(() => {
    setState(prev => {
      const newMuted = !prev.isMuted
      if (newMuted) {
        stopPolling()
      }
      return { ...prev, isMuted: newMuted }
    })
  }, [stopPolling])

  /**
   * Mark chat as visible/focused (clears unread count)
   */
  const markAsRead = useCallback(() => {
    isVisibleRef.current = true
    setState(prev => ({ ...prev, unreadCount: 0 }))
  }, [])

  /**
   * Mark chat as hidden/unfocused
   */
  const markAsHidden = useCallback(() => {
    isVisibleRef.current = false
  }, [])

  /**
   * Clear all messages (for new game)
   */
  const clearMessages = useCallback(() => {
    lastMessageTimeRef.current = 0
    setState(prev => ({
      ...prev,
      messages: [],
      unreadCount: 0,
      error: null,
    }))
  }, [])

  /**
   * Send a quick reaction message
   */
  const sendReaction = useCallback((reaction: string) => {
    return sendMessage(reaction)
  }, [sendMessage])

  return {
    ...state,
    sendMessage,
    sendReaction,
    toggleMute,
    markAsRead,
    markAsHidden,
    clearMessages,
  }
}

// Quick reaction presets
export const QUICK_REACTIONS = [
  { label: 'GG', message: 'Good game!' },
  { label: 'Nice!', message: 'Nice move!' },
  { label: 'GL', message: 'Good luck!' },
  { label: 'Rematch?', message: 'Want to play again?' },
] as const
