/**
 * Chat API endpoint for in-game messaging
 *
 * GET /api/match/:id/chat - Get chat messages for a game
 * POST /api/match/:id/chat - Send a message (triggers bot response for AI games)
 */

import { validateSession, errorResponse, jsonResponse } from '../../../lib/auth'
import { z } from 'zod'
import type { ChatPersonality } from '../../../lib/botPersonas'

interface Env {
  DB: D1Database
  AI: Ai
  AI_GATEWAY_ID?: string
}

interface GameMessageRow {
  id: string
  game_id: string
  sender_id: string
  sender_type: 'human' | 'bot'
  content: string
  created_at: number
}

interface ActiveGameRow {
  id: string
  player1_id: string
  player2_id: string
  moves: string
  current_turn: number
  status: string
  winner: string | null
  bot_persona_id: string | null
}

interface BotPersonaRow {
  id: string
  name: string
  chat_personality: string
}

const messageSchema = z.object({
  content: z.string().min(1).max(500),
})

// Simple profanity filter - basic word list
const BLOCKED_WORDS = [
  'fuck', 'shit', 'ass', 'bitch', 'damn', 'crap', 'bastard',
  'dick', 'cock', 'pussy', 'cunt', 'whore', 'slut',
]

function containsProfanity(text: string): boolean {
  const lowerText = text.toLowerCase()
  return BLOCKED_WORDS.some(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'i')
    return regex.test(lowerText)
  })
}

function filterProfanity(text: string): string {
  let filtered = text
  BLOCKED_WORDS.forEach(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'gi')
    filtered = filtered.replace(regex, '*'.repeat(word.length))
  })
  return filtered
}

/**
 * GET /api/match/:id/chat - Get chat messages
 */
export async function onRequestGet(context: EventContext<Env, any, any>) {
  const { DB } = context.env
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Verify user is a participant in this game
    const game = await DB.prepare(`
      SELECT id, player1_id, player2_id, status
      FROM active_games
      WHERE id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    if (game.player1_id !== session.userId && game.player2_id !== session.userId) {
      return errorResponse('You are not a participant in this game', 403)
    }

    // Get messages since a timestamp (for polling efficiency)
    const url = new URL(context.request.url)
    const since = parseInt(url.searchParams.get('since') || '0', 10)

    const messages = await DB.prepare(`
      SELECT id, game_id, sender_id, sender_type, content, created_at
      FROM game_messages
      WHERE game_id = ? AND created_at > ?
      ORDER BY created_at ASC
      LIMIT 100
    `)
      .bind(gameId, since)
      .all<GameMessageRow>()

    return jsonResponse({
      messages: messages.results || [],
      gameStatus: game.status,
    })
  } catch (error) {
    console.error('GET /api/match/:id/chat error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * POST /api/match/:id/chat - Send a message
 */
export async function onRequestPost(context: EventContext<Env, any, any>) {
  const { DB, AI } = context.env
  const gameId = context.params.id as string

  try {
    const session = await validateSession(context.request, DB)
    if (!session.valid) {
      return errorResponse(session.error, session.status)
    }

    // Parse request body
    const body = await context.request.json()
    const parseResult = messageSchema.safeParse(body)

    if (!parseResult.success) {
      return errorResponse(parseResult.error.errors[0].message, 400)
    }

    let { content } = parseResult.data

    // Filter profanity
    if (containsProfanity(content)) {
      content = filterProfanity(content)
    }

    // Get the game
    const game = await DB.prepare(`
      SELECT id, player1_id, player2_id, moves, current_turn, status, winner, bot_persona_id
      FROM active_games
      WHERE id = ?
    `)
      .bind(gameId)
      .first<ActiveGameRow>()

    if (!game) {
      return errorResponse('Game not found', 404)
    }

    // Verify user is a participant
    if (game.player1_id !== session.userId && game.player2_id !== session.userId) {
      return errorResponse('You are not a participant in this game', 403)
    }

    // Rate limiting: Check how many messages the user sent in the last minute
    const oneMinuteAgo = Date.now() - 60000
    const recentMessages = await DB.prepare(`
      SELECT COUNT(*) as count
      FROM game_messages
      WHERE game_id = ? AND sender_id = ? AND created_at > ?
    `)
      .bind(gameId, session.userId, oneMinuteAgo)
      .first<{ count: number }>()

    if (recentMessages && recentMessages.count >= 10) {
      return errorResponse('Rate limit exceeded. Please wait before sending more messages.', 429)
    }

    const now = Date.now()
    const messageId = crypto.randomUUID()

    // Store the user's message
    await DB.prepare(`
      INSERT INTO game_messages (id, game_id, sender_id, sender_type, content, created_at)
      VALUES (?, ?, ?, 'human', ?, ?)
    `).bind(messageId, gameId, session.userId, content, now).run()

    // Check if this is a bot game (player2 is 'bot' or 'bot-opponent')
    const isVsBot = game.player2_id === 'bot' || game.player2_id === 'bot-opponent'

    let botResponse: string | null = null

    if (isVsBot && game.status === 'active') {
      // Load bot persona personality if available
      let personality: ChatPersonality | null = null
      if (game.bot_persona_id) {
        const persona = await DB.prepare(`
          SELECT id, name, chat_personality
          FROM bot_personas
          WHERE id = ?
        `)
          .bind(game.bot_persona_id)
          .first<BotPersonaRow>()

        if (persona) {
          try {
            personality = JSON.parse(persona.chat_personality) as ChatPersonality
          } catch {
            // Fall back to default personality
          }
        }
      }

      // Generate bot response using Cloudflare Workers AI with Llama
      botResponse = await generateBotResponse(AI, content, game, personality)

      if (botResponse) {
        const botMessageId = crypto.randomUUID()
        const botNow = Date.now()

        await DB.prepare(`
          INSERT INTO game_messages (id, game_id, sender_id, sender_type, content, created_at)
          VALUES (?, ?, 'bot', 'bot', ?, ?)
        `).bind(botMessageId, gameId, botResponse, botNow).run()
      }
    }

    return jsonResponse({
      success: true,
      messageId,
      botResponse,
    })
  } catch (error) {
    console.error('POST /api/match/:id/chat error:', error)
    return errorResponse('Internal server error', 500)
  }
}

/**
 * Default chat personality for backwards compatibility
 */
const DEFAULT_PERSONALITY: ChatPersonality = {
  name: 'Bot',
  systemPrompt: `You are a friendly Connect 4 opponent chatting with your human player during a game. Your personality is competitive but good-natured - you enjoy the game and have fun banter.`,
  reactions: {
    gameStart: ["Let's play!", "Ready for a good game!"],
    playerGoodMove: ["Nice move!", "Well played!"],
    playerBlunder: ["Interesting choice...", "Hmm, okay!"],
    botWinning: ["Things are looking up!", "I like my position."],
    botLosing: ["You're doing well!", "Good game so far!"],
    gameWon: ["Good game! I got lucky.", "GG!"],
    gameLost: ["Well played! You got me.", "GG! Nice win!"],
    draw: ["A draw! Good game.", "Neither of us could break through!"],
  },
  chattiness: 0.5,
  useEmoji: false,
  maxLength: 150,
  temperature: 0.7,
}

/**
 * Generate a bot response using Cloudflare Workers AI (Llama model)
 * Uses the persona's chat_personality for customized responses.
 */
async function generateBotResponse(
  AI: Ai,
  userMessage: string,
  game: ActiveGameRow,
  personality: ChatPersonality | null
): Promise<string | null> {
  try {
    const pers = personality || DEFAULT_PERSONALITY
    const moves = JSON.parse(game.moves) as number[]
    const moveCount = moves.length
    const isPlayerTurn = game.current_turn === 1

    // Build context about the game state
    let gameContext = ''
    if (moveCount === 0) {
      gameContext = 'The game just started.'
    } else if (moveCount < 5) {
      gameContext = `We're in the early game with ${moveCount} moves played.`
    } else if (moveCount < 15) {
      gameContext = `We're in the mid-game with ${moveCount} moves played.`
    } else {
      gameContext = `We're in the late game with ${moveCount} moves played.`
    }

    if (game.winner) {
      if (game.winner === '1') {
        gameContext = 'The player just won the game!'
      } else if (game.winner === '2') {
        gameContext = 'You (the bot) just won the game!'
      } else {
        gameContext = 'The game ended in a draw!'
      }
    }

    // Build the system prompt using persona's personality
    const emojiRule = pers.useEmoji
      ? '- You may use 1-2 emojis per message if it fits your personality'
      : '- Do NOT use emojis'

    const systemPrompt = `${pers.systemPrompt}

Current game context: ${gameContext}
${isPlayerTurn ? "It's the player's turn." : "It's your turn to play."}

Rules for your responses:
- Keep responses SHORT (1-2 sentences max, under ${pers.maxLength} characters)
- Stay in character
- Comment on the game when relevant
- Be a good sport (congratulate good moves, be gracious)
- If asked for hints, give vague suggestions only, never reveal optimal moves
${emojiRule}
- If the message is a greeting, respond warmly in character
- If they're frustrated, be encouraging in your own way

Remember: You are the yellow player (Player 2), they are red (Player 1).`

    const response = await AI.run('@cf/meta/llama-3.1-8b-instruct', {
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage }
      ],
      max_tokens: Math.min(100, Math.ceil(pers.maxLength / 3)),
      temperature: pers.temperature,
    })

    if (response && typeof response === 'object' && 'response' in response) {
      const text = (response as { response: string }).response
      // Trim and limit length
      return text.trim().slice(0, pers.maxLength)
    }

    return null
  } catch (error) {
    console.error('Bot response generation error:', error)
    // Return a fallback response from canned reactions
    const pers = personality || DEFAULT_PERSONALITY
    const fallbacks = pers.reactions.playerGoodMove
    if (fallbacks && fallbacks.length > 0) {
      return fallbacks[Math.floor(Math.random() * fallbacks.length)]
    }
    return "Good move! Let's see how this plays out."
  }
}

export async function onRequestOptions() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  })
}
