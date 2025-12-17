/**
 * Debug Chat Page - Test chat interface in isolation
 *
 * Designed for headless browser testing:
 * - Plain textarea with debug logs for easy scraping
 * - Data attributes for element selection
 * - Debug login (no real credentials needed)
 * - Debug game creation
 * - Streams all events to a single log textarea
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { Button } from '../components/ui/button'
import { Input } from '../components/ui/input'
import { useGameChat, QUICK_REACTIONS, type ChatMessage } from '../hooks/useGameChat'
import { useAuth } from '../contexts/AuthContext'

export default function DebugChatPage() {
  const { user, isAuthenticated } = useAuth()
  const [gameId, setGameId] = useState('')
  const [isBot, setIsBot] = useState(true)
  const [isActive, setIsActive] = useState(false)
  const [moveCount, setMoveCount] = useState(0)
  const [debugLog, setDebugLog] = useState('')
  const [messageInput, setMessageInput] = useState('')
  const [messageHistory, setMessageHistory] = useState<ChatMessage[]>([])
  const [isLoggingIn, setIsLoggingIn] = useState(false)
  const [isCreatingGame, setIsCreatingGame] = useState(false)
  const logTextareaRef = useRef<HTMLTextAreaElement>(null)

  const chat = useGameChat(isActive ? gameId : null, isActive)

  // Append to debug log (plain text for headless browser scraping)
  const log = useCallback((level: string, message: string, data?: unknown) => {
    const timestamp = new Date().toISOString()
    let line = `[${timestamp}] [${level.toUpperCase()}] ${message}`
    if (data !== undefined) {
      line += `\n  DATA: ${JSON.stringify(data)}`
    }
    setDebugLog(prev => prev + line + '\n')
  }, [])

  // Auto-scroll log textarea
  useEffect(() => {
    if (logTextareaRef.current) {
      logTextareaRef.current.scrollTop = logTextareaRef.current.scrollHeight
    }
  }, [debugLog])

  // Log auth state on mount
  useEffect(() => {
    log('info', `Auth state: ${isAuthenticated ? 'authenticated' : 'not authenticated'}`, {
      userId: user?.id,
      username: user?.username,
    })
  }, [isAuthenticated, user, log])

  // Track chat state changes
  useEffect(() => {
    log('state', 'Chat state updated', {
      isActive,
      isMuted: chat.isMuted,
      isSending: chat.isSending,
      isRequestingReaction: chat.isRequestingReaction,
      messageCount: chat.messages.length,
      unreadCount: chat.unreadCount,
      error: chat.error,
    })
  }, [
    isActive,
    chat.isMuted,
    chat.isSending,
    chat.isRequestingReaction,
    chat.messages.length,
    chat.unreadCount,
    chat.error,
    log,
  ])

  // Track message changes
  useEffect(() => {
    if (chat.messages.length > 0) {
      const newMessages = chat.messages.filter(
        m => !messageHistory.find(h => h.id === m.id)
      )
      if (newMessages.length > 0) {
        for (const msg of newMessages) {
          log('message', `New message received`, {
            id: msg.id,
            sender_type: msg.sender_type,
            sender_id: msg.sender_id,
            content: msg.content,
            created_at: msg.created_at,
          })
        }

        // Check for duplicates
        const seen = new Set<string>()
        const duplicates: string[] = []
        for (const m of chat.messages) {
          if (seen.has(m.id)) {
            duplicates.push(m.id)
          }
          seen.add(m.id)
        }
        if (duplicates.length > 0) {
          log('warn', `DUPLICATE MESSAGES DETECTED!`, { duplicateIds: duplicates })
        }
      }
      setMessageHistory([...chat.messages])
    }
  }, [chat.messages, messageHistory, log])

  // Track errors
  useEffect(() => {
    if (chat.error) {
      log('error', `Chat error: ${chat.error}`)
    }
  }, [chat.error, log])

  // Debug login handler
  const handleDebugLogin = async () => {
    setIsLoggingIn(true)
    log('info', 'Attempting debug login...')

    try {
      const response = await fetch('/api/debug/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })

      const data = await response.json()

      if (!response.ok) {
        log('error', 'Debug login failed', data)
        return
      }

      log('success', 'Debug login successful', { userId: data.user.id, username: data.user.username })

      // Store the session token and reload to update auth context
      localStorage.setItem('makefour_session_token', data.session_token)
      window.location.reload()
    } catch (error) {
      log('error', 'Debug login error', { error: String(error) })
    } finally {
      setIsLoggingIn(false)
    }
  }

  // Debug game creation handler
  const handleCreateDebugGame = async () => {
    if (!isAuthenticated) {
      log('error', 'Must be logged in to create a game')
      return
    }

    setIsCreatingGame(true)
    log('info', 'Creating debug game...')

    try {
      const token = localStorage.getItem('makefour_session_token')
      const response = await fetch('/api/debug/game', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
      })

      const data = await response.json()

      if (!response.ok) {
        log('error', 'Debug game creation failed', data)
        return
      }

      log('success', 'Debug game created', { gameId: data.gameId, botName: data.botName })
      setGameId(data.gameId)
    } catch (error) {
      log('error', 'Debug game creation error', { error: String(error) })
    } finally {
      setIsCreatingGame(false)
    }
  }

  const handleConnect = () => {
    if (!gameId.trim()) {
      log('error', 'Game ID is required')
      return
    }
    log('info', `Connecting to game: ${gameId}`, { isBot })
    setIsActive(true)
    setMessageHistory([])
  }

  const handleDisconnect = () => {
    log('info', 'Disconnecting from game')
    setIsActive(false)
    chat.clearMessages()
    setMessageHistory([])
  }

  const handleSendMessage = async () => {
    const content = messageInput.trim()
    if (!content) return

    log('info', `Sending message: "${content}"`)
    setMessageInput('')

    const success = await chat.sendMessage(content)
    if (success) {
      log('success', 'Message sent successfully')
    } else {
      log('error', 'Failed to send message', { error: chat.error })
    }
  }

  const handleTriggerBotReaction = () => {
    const newMoveCount = moveCount + 1
    setMoveCount(newMoveCount)
    log('info', `Triggering bot reaction for move ${newMoveCount}`)
    chat.triggerBotReaction(newMoveCount, true)
  }

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 p-4" data-testid="debug-chat-page">
      <div className="max-w-4xl mx-auto space-y-4">
        <h1 className="text-2xl font-bold" data-testid="page-title">Debug: Chat Interface</h1>

        {/* Status Bar */}
        <div
          className="p-3 rounded bg-white dark:bg-gray-800 border text-sm font-mono"
          data-testid="status-bar"
        >
          <div data-testid="auth-status">
            AUTH: {isAuthenticated ? `YES (${user?.username || user?.email} / ${user?.id})` : 'NO'}
          </div>
          <div data-testid="connection-status">
            CONNECTION: {isActive ? `ACTIVE (${gameId})` : 'INACTIVE'}
          </div>
          <div data-testid="chat-status">
            CHAT: messages={chat.messages.length} muted={String(chat.isMuted)} sending={String(chat.isSending)} error={chat.error || 'none'}
          </div>
        </div>

        {/* Debug Actions */}
        <div className="p-4 rounded bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 space-y-3">
          <h2 className="font-semibold text-yellow-800 dark:text-yellow-200">Debug Actions</h2>
          <div className="flex gap-2 flex-wrap">
            {!isAuthenticated ? (
              <Button
                onClick={handleDebugLogin}
                disabled={isLoggingIn}
                data-testid="debug-login-btn"
                className="bg-yellow-600 hover:bg-yellow-700"
              >
                {isLoggingIn ? 'Logging in...' : 'Login as Debug User'}
              </Button>
            ) : (
              <>
                <Button
                  onClick={handleCreateDebugGame}
                  disabled={isCreatingGame || isActive}
                  data-testid="create-game-btn"
                  className="bg-green-600 hover:bg-green-700"
                >
                  {isCreatingGame ? 'Creating...' : 'Create Debug Game'}
                </Button>
                <span className="text-sm text-gray-600 dark:text-gray-400 self-center">
                  Logged in as {user?.username || user?.email}
                </span>
              </>
            )}
          </div>
        </div>

        {/* Controls */}
        <div className="p-4 rounded bg-white dark:bg-gray-800 border space-y-3">
          <div className="flex gap-2 items-end flex-wrap">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium mb-1">Game ID</label>
              <Input
                value={gameId}
                onChange={e => setGameId(e.target.value)}
                placeholder="Enter game ID or create one above"
                disabled={isActive}
                data-testid="game-id-input"
              />
            </div>
            <label className="flex items-center gap-2 text-sm pb-2">
              <input
                type="checkbox"
                checked={isBot}
                onChange={e => setIsBot(e.target.checked)}
                disabled={isActive}
                data-testid="is-bot-checkbox"
              />
              Bot Game
            </label>
            {!isActive ? (
              <Button onClick={handleConnect} disabled={!gameId.trim()} data-testid="connect-btn">
                Connect
              </Button>
            ) : (
              <Button onClick={handleDisconnect} variant="destructive" data-testid="disconnect-btn">
                Disconnect
              </Button>
            )}
          </div>

          {/* Message Input */}
          <div className="flex gap-2">
            <Input
              value={messageInput}
              onChange={e => setMessageInput(e.target.value)}
              placeholder="Type a message..."
              disabled={!isActive || chat.isSending}
              onKeyDown={e => e.key === 'Enter' && handleSendMessage()}
              data-testid="message-input"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!isActive || chat.isSending || !messageInput.trim()}
              data-testid="send-btn"
            >
              Send
            </Button>
          </div>

          {/* Quick Actions */}
          <div className="flex gap-2 flex-wrap">
            {QUICK_REACTIONS.map(r => (
              <Button
                key={r.label}
                size="sm"
                variant="outline"
                onClick={() => {
                  setMessageInput(r.message)
                }}
                disabled={!isActive}
                data-testid={`quick-${r.label.toLowerCase()}`}
              >
                {r.label}
              </Button>
            ))}
            <Button
              size="sm"
              variant="outline"
              onClick={handleTriggerBotReaction}
              disabled={!isActive || !isBot || chat.isRequestingReaction}
              data-testid="trigger-bot-btn"
            >
              Trigger Bot (move={moveCount + 1})
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                chat.toggleMute()
                log('info', `Toggled mute: ${!chat.isMuted}`)
              }}
              disabled={!isActive}
              data-testid="mute-btn"
            >
              {chat.isMuted ? 'Unmute' : 'Mute'}
            </Button>
          </div>
        </div>

        {/* Debug Log Textarea - Main output for headless browser */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <label className="font-semibold">Debug Log (for headless browser)</label>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setDebugLog('')}
              data-testid="clear-log-btn"
            >
              Clear
            </Button>
          </div>
          <textarea
            ref={logTextareaRef}
            value={debugLog}
            readOnly
            className="w-full h-80 p-3 font-mono text-xs bg-black text-green-400 rounded border-0 resize-none"
            data-testid="debug-log"
            id="debug-log"
          />
        </div>

        {/* Messages JSON - for programmatic access */}
        <div className="space-y-2">
          <label className="font-semibold">Messages JSON</label>
          <textarea
            value={JSON.stringify(chat.messages, null, 2)}
            readOnly
            className="w-full h-48 p-3 font-mono text-xs bg-gray-50 dark:bg-gray-800 rounded border resize-none"
            data-testid="messages-json"
            id="messages-json"
          />
        </div>

        {/* Instructions */}
        <div className="p-4 rounded bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 text-sm">
          <h3 className="font-semibold mb-2">Headless Browser Testing</h3>
          <ol className="list-decimal list-inside space-y-1 text-gray-700 dark:text-gray-300">
            <li>Click "Login as Debug User" to authenticate</li>
            <li>Click "Create Debug Game" to create a test game</li>
            <li>Click "Connect" to connect to the game's chat</li>
            <li>Send messages and watch the debug log</li>
            <li>Click "Trigger Bot" to simulate moves and test bot reactions</li>
          </ol>
          <h4 className="font-semibold mt-4 mb-2">Element Selectors</h4>
          <ul className="space-y-1 text-gray-700 dark:text-gray-300">
            <li><code>data-testid="debug-login-btn"</code> - Login button</li>
            <li><code>data-testid="create-game-btn"</code> - Create game button</li>
            <li><code>data-testid="connect-btn"</code> - Connect button</li>
            <li><code>data-testid="debug-log"</code> - Main log textarea</li>
            <li><code>data-testid="messages-json"</code> - Raw messages JSON</li>
          </ul>
        </div>
      </div>
    </div>
  )
}
