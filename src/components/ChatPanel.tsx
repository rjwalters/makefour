/**
 * ChatPanel component for in-game messaging
 *
 * Features:
 * - Message display with human/bot distinction
 * - Message input with send button
 * - Quick reactions (GG, Nice!, GL, Rematch?)
 * - Mute/unmute toggle
 * - Collapsible on mobile
 * - Unread message indicator
 */

import { useState, useRef, useEffect, type FormEvent } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { cn } from '@/lib/utils'
import { useGameChat, QUICK_REACTIONS, type ChatMessage } from '../hooks/useGameChat'
import BotAvatar from './BotAvatar'

interface ChatPanelProps {
  gameId: string | null
  isActive?: boolean
  isBot?: boolean
  playerNumber?: 1 | 2
  className?: string
  /** Current move count - used to trigger bot reactions after player moves */
  moveCount?: number
  /** Bot avatar URL (emoji or image URL) for displaying next to bot messages */
  botAvatarUrl?: string | null
  /** Bot name for display in chat */
  botName?: string
  /** Current user's ID for identifying own messages */
  userId?: string
  /** Opponent's display name */
  opponentName?: string
}

export default function ChatPanel({
  gameId,
  isActive = true,
  isBot = false,
  className,
  moveCount = 0,
  botAvatarUrl,
  botName = 'Bot',
  userId,
  opponentName = 'Opponent',
}: ChatPanelProps) {
  const {
    messages,
    isSending,
    error,
    isMuted,
    unreadCount,
    sendMessage,
    sendReaction,
    toggleMute,
    markAsRead,
    markAsHidden,
    triggerBotReaction,
    resetMoveTracking,
  } = useGameChat(gameId, isActive)

  const [inputValue, setInputValue] = useState('')
  const [isCollapsed, setIsCollapsed] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (!isCollapsed) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, isCollapsed])

  // Trigger bot reactions when move count changes (for bot games only)
  useEffect(() => {
    if (isBot && moveCount > 0) {
      triggerBotReaction(moveCount, true)
    }
  }, [isBot, moveCount, triggerBotReaction])

  // Reset move tracking when game changes
  useEffect(() => {
    resetMoveTracking()
  }, [gameId, resetMoveTracking])

  // Mark as read when panel is expanded
  useEffect(() => {
    if (!isCollapsed) {
      markAsRead()
    } else {
      markAsHidden()
    }
  }, [isCollapsed, markAsRead, markAsHidden])

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!inputValue.trim() || isSending) return

    const success = await sendMessage(inputValue)
    if (success) {
      setInputValue('')
    }
  }

  const handleQuickReaction = async (message: string) => {
    await sendReaction(message)
  }

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  }

  const getSenderName = (message: ChatMessage) => {
    if (message.sender_type === 'bot') {
      return botName
    }
    // Check if this is the current player's message
    if (message.sender_id === 'self' || message.sender_id === userId) {
      return 'You'
    }
    return opponentName
  }

  const isOwnMessage = (message: ChatMessage) => {
    if (message.sender_type === 'bot') {
      return false
    }
    return message.sender_id === 'self' || message.sender_id === userId
  }

  if (!gameId) {
    return null
  }

  return (
    <div
      className={cn(
        'flex flex-col bg-white dark:bg-gray-800 rounded-lg border shadow-sm',
        isCollapsed ? 'h-12' : 'h-64',
        className
      )}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2 border-b cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">
            Chat {isBot && <span className="text-xs text-muted-foreground">(vs Bot)</span>}
          </span>
          {unreadCount > 0 && isCollapsed && (
            <span className="bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full min-w-[20px] text-center">
              {unreadCount}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={(e) => {
              e.stopPropagation()
              toggleMute()
            }}
            title={isMuted ? 'Unmute chat' : 'Mute chat'}
          >
            {isMuted ? (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path d="M9.547 3.062A.75.75 0 0110.5 3.75v12.5a.75.75 0 01-1.264.546L5.203 13H2.667a.75.75 0 01-.7-.48A6.985 6.985 0 011.5 10c0-.887.165-1.737.468-2.52a.75.75 0 01.7-.48h2.535l4.033-3.796a.75.75 0 01.811-.142zM13.28 7.22a.75.75 0 10-1.06 1.06L13.94 10l-1.72 1.72a.75.75 0 001.06 1.06L15 11.06l1.72 1.72a.75.75 0 101.06-1.06L16.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L15 8.94l-1.72-1.72z" />
              </svg>
            ) : (
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                <path d="M10.5 3.75a.75.75 0 00-1.264-.546L5.203 7H2.667a.75.75 0 00-.7.48A6.985 6.985 0 001.5 10c0 .887.165 1.737.468 2.52.111.29.39.48.7.48h2.535l4.033 3.796a.75.75 0 001.264-.546V3.75zM14.5 10a4.5 4.5 0 01-1.318 3.182.75.75 0 101.06 1.06A6 6 0 0016 10a6 6 0 00-1.758-4.243.75.75 0 00-1.06 1.061A4.5 4.5 0 0114.5 10z" />
              </svg>
            )}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0"
            onClick={(e) => {
              e.stopPropagation()
              setIsCollapsed(!isCollapsed)
            }}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className={cn('w-4 h-4 transition-transform', isCollapsed && 'rotate-180')}
            >
              <path fillRule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clipRule="evenodd" />
            </svg>
          </Button>
        </div>
      </div>

      {/* Chat content - hidden when collapsed */}
      {!isCollapsed && (
        <>
          {/* Messages area */}
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-2">
            {isMuted ? (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                Chat is muted
              </div>
            ) : messages.length === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground text-sm">
                {isBot ? 'Say hi to the bot!' : 'Send a message to your opponent'}
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    'flex max-w-[85%]',
                    isOwnMessage(message) ? 'ml-auto flex-row-reverse' : 'flex-row',
                    'gap-2'
                  )}
                >
                  {/* Bot avatar */}
                  {message.sender_type === 'bot' && (
                    <BotAvatar
                      avatarUrl={botAvatarUrl ?? null}
                      name={botName}
                      size="xs"
                      className="flex-shrink-0 mt-0.5"
                    />
                  )}
                  <div className="flex flex-col">
                    <div
                      className={cn(
                        'rounded-lg px-3 py-1.5 text-sm',
                        message.sender_type === 'bot'
                          ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-900 dark:text-purple-100'
                          : isOwnMessage(message)
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted'
                      )}
                    >
                      {message.content}
                    </div>
                    <span className="text-xs text-muted-foreground mt-0.5">
                      {getSenderName(message)} · {formatTime(message.created_at)}
                    </span>
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick reactions */}
          {!isMuted && (
            <div className="flex gap-1 px-3 py-1 border-t">
              {QUICK_REACTIONS.map((reaction) => (
                <Button
                  key={reaction.label}
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={() => handleQuickReaction(reaction.message)}
                  disabled={isSending}
                >
                  {reaction.label}
                </Button>
              ))}
            </div>
          )}

          {/* Input area */}
          {!isMuted && (
            <form onSubmit={handleSubmit} className="flex gap-2 px-3 py-2 border-t">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Type a message..."
                className="h-8 text-sm"
                maxLength={500}
                disabled={isSending}
              />
              <Button
                type="submit"
                size="sm"
                className="h-8 px-3"
                disabled={!inputValue.trim() || isSending}
              >
                {isSending ? (
                  <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                    <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
                  </svg>
                )}
              </Button>
            </form>
          )}

          {/* Error message */}
          {error && (
            <div className="px-3 py-1 text-xs text-red-500 border-t">
              {error}
            </div>
          )}
        </>
      )}
    </div>
  )
}
