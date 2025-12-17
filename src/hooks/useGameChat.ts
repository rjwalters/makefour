/**
 * Custom React hook for managing in-game chat
 *
 * Handles:
 * - Polling for new messages
 * - Sending messages
 * - Bot response handling
 * - Mute/unmute functionality
 * - Proactive bot reactions to game moves
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
  isRequestingReaction: boolean
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
    isRequestingReaction: false,
  })

  const pollRef = useRef<NodeJS.Timeout | null>(null)
  const isMountedRef = useRef(true)
  const lastMessageTimeRef = useRef(0)
  const isVisibleRef = useRef(true) // Track if chat panel is visible
  const lastMoveCountRef = useRef(0) // Track move count for bot reactions
  const isSendingRef = useRef(false) // Additional safeguard against double sends
  const lastSentContentRef = useRef<string | null>(null) // Track last sent message

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

        console.log('[Chat Debug] fetchMessages received:', response.messages.map(m => ({ id: m.id, content: m.content })))

        setState(prev => {
          // Merge new messages, avoiding duplicates
          const existingIds = new Set(prev.messages.map(m => m.id))
          const newMessages = response.messages.filter(m => !existingIds.has(m.id))

          console.log('[Chat Debug] fetchMessages dedup - existing IDs:', Array.from(existingIds))
          console.log('[Chat Debug] fetchMessages dedup - new messages after filter:', newMessages.map(m => m.id))

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

    const trimmedContent = content.trim()

    // Debug: Log send attempt
    console.log('[Chat Debug] sendMessage called:', { content: trimmedContent, gameId, isSendingRef: isSendingRef.current })

    // Safeguard: Prevent double sends using ref (more reliable than state)
    if (isSendingRef.current) {
      console.log('[Chat Debug] BLOCKED - already sending (ref check)')
      return false
    }

    // Safeguard: Prevent sending exact same message twice in quick succession
    if (lastSentContentRef.current === trimmedContent) {
      console.log('[Chat Debug] BLOCKED - same content as last message')
      return false
    }

    isSendingRef.current = true
    lastSentContentRef.current = trimmedContent

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

      console.log('[Chat Debug] Adding optimistic message:', { id: response.messageId, content: content.trim() })

      setState(prev => {
        console.log('[Chat Debug] Current message count before add:', prev.messages.length)
        console.log('[Chat Debug] Existing message IDs:', prev.messages.map(m => m.id))
        return {
          ...prev,
          messages: [...prev.messages, userMessage],
          isSending: false,
        }
      })

      // Reset the sending ref after successful send
      isSendingRef.current = false

      // Clear lastSentContent after a delay to allow same message later
      setTimeout(() => {
        lastSentContentRef.current = null
      }, 1000)

      // Trigger immediate poll to get any bot response
      setTimeout(fetchMessages, 100)

      return true
    } catch (error) {
      // Always reset the sending ref on error
      isSendingRef.current = false
      lastSentContentRef.current = null

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

  /**
   * Trigger a proactive bot reaction after a player move.
   * Should be called after the player makes a move in a bot game.
   * The bot may or may not respond based on game analysis.
   *
   * @param moveCount - Current total move count in the game
   * @param isVsBot - Whether this is a bot game
   */
  const triggerBotReaction = useCallback(async (moveCount: number, isVsBot: boolean) => {
    // Skip if not a bot game, muted, or no new moves
    if (!gameId || !isVsBot || state.isMuted) return

    // Only trigger for new moves (avoid duplicate calls)
    if (moveCount <= lastMoveCountRef.current) return
    lastMoveCountRef.current = moveCount

    // Don't trigger if already requesting
    if (state.isRequestingReaction) return

    setState(prev => ({ ...prev, isRequestingReaction: true }))

    try {
      const response = await apiCall<{
        message: string | null
        messageId?: string
        reason: string
      }>(`/api/match/${gameId}/bot-reaction`, {
        method: 'POST',
      })

      if (!isMountedRef.current) return

      // If bot generated a message, poll to get it
      if (response.message) {
        // Small delay then poll to pick up the new message
        setTimeout(fetchMessages, 200)
      }

      setState(prev => ({ ...prev, isRequestingReaction: false }))
    } catch (error) {
      // Silently fail - bot reactions are nice-to-have, not critical
      console.error('Bot reaction error:', error)
      if (isMountedRef.current) {
        setState(prev => ({ ...prev, isRequestingReaction: false }))
      }
    }
  }, [apiCall, gameId, state.isMuted, state.isRequestingReaction, fetchMessages])

  /**
   * Reset move count tracking (for new games)
   */
  const resetMoveTracking = useCallback(() => {
    lastMoveCountRef.current = 0
  }, [])

  return {
    ...state,
    sendMessage,
    sendReaction,
    toggleMute,
    markAsRead,
    markAsHidden,
    clearMessages,
    triggerBotReaction,
    resetMoveTracking,
  }
}

// Quick reaction presets
export const QUICK_REACTIONS = [
  { label: 'GG', message: 'Good game!' },
  { label: 'Nice!', message: 'Nice move!' },
  { label: 'GL', message: 'Good luck!' },
  { label: 'Rematch?', message: 'Want to play again?' },
] as const
