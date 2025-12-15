/**
 * Bot Personas - Definitions and utilities for AI opponents
 *
 * Each persona has a unique personality, playing style, and skill level.
 * The ai_config maps to the minimax engine parameters.
 */

export interface BotPersona {
  id: string
  name: string
  description: string
  avatar_url: string | null
  ai_engine: 'minimax' | 'aggressive-minimax' | 'deep-minimax' | 'neural'
  ai_config: {
    searchDepth: number
    errorRate: number
    timeMultiplier?: number
    /** Custom evaluation weights for configurable engines */
    evalWeights?: {
      ownThreats?: number
      opponentThreats?: number
      centerControl?: number
      doubleThreats?: number
    }
    /** Use transposition table (for deep-minimax) */
    useTranspositionTable?: boolean
  }
  chat_personality: {
    style: string
    greeting: string
    onWin: string
    onLose: string
    onGoodMove: string
    onBadMove: string
    tauntFrequency: number
  }
  play_style: 'aggressive' | 'defensive' | 'balanced' | 'tricky' | 'adaptive'
  base_elo: number
}

/**
 * Default bot personas to seed the database
 */
export const DEFAULT_BOT_PERSONAS: BotPersona[] = [
  {
    id: 'rookie',
    name: 'Rookie',
    description: 'A friendly beginner still learning the ropes. Makes lots of mistakes but never gives up!',
    avatar_url: null,
    ai_engine: 'minimax',
    ai_config: {
      searchDepth: 2,
      errorRate: 0.35,
      timeMultiplier: 0.1,
    },
    chat_personality: {
      style: 'enthusiastic',
      greeting: "Hi! I'm still learning, so go easy on me!",
      onWin: "Wow, I won! I'm getting better!",
      onLose: "Good game! I'll get you next time!",
      onGoodMove: 'Nice move! I need to pay more attention.',
      onBadMove: "Oops, that wasn't my best move...",
      tauntFrequency: 0.1,
    },
    play_style: 'balanced',
    base_elo: 700,
  },
  {
    id: 'rusty',
    name: 'Rusty',
    description: "An old-timer getting back into the game. Solid fundamentals but occasionally rusty.",
    avatar_url: null,
    ai_engine: 'minimax',
    ai_config: {
      searchDepth: 3,
      errorRate: 0.25,
      timeMultiplier: 0.2,
    },
    chat_personality: {
      style: 'nostalgic',
      greeting: "Ah, this takes me back. Let's see if I still got it.",
      onWin: "Still got the old magic!",
      onLose: "Well played. The new generation is sharp.",
      onGoodMove: "That reminds me of a game I played years ago...",
      onBadMove: "Hmm, I used to be sharper than that.",
      tauntFrequency: 0.15,
    },
    play_style: 'defensive',
    base_elo: 900,
  },
  {
    id: 'nova',
    name: 'Nova',
    description: 'A promising player with flashes of brilliance. Occasionally makes mistakes under pressure.',
    avatar_url: null,
    ai_engine: 'minimax',
    ai_config: {
      searchDepth: 4,
      errorRate: 0.15,
      timeMultiplier: 0.3,
    },
    chat_personality: {
      style: 'confident',
      greeting: "Ready to shine! Let's have a great game.",
      onWin: "That's what I'm talking about!",
      onLose: "Impressive! I'll study that game.",
      onGoodMove: "Okay, I see you!",
      onBadMove: "Wait, that wasn't the plan...",
      tauntFrequency: 0.2,
    },
    play_style: 'aggressive',
    base_elo: 1100,
  },
  {
    id: 'scholar',
    name: 'Scholar',
    description: 'A methodical player who has studied every opening. Rarely makes mistakes.',
    avatar_url: null,
    ai_engine: 'minimax',
    ai_config: {
      searchDepth: 6,
      errorRate: 0.08,
      timeMultiplier: 0.5,
    },
    chat_personality: {
      style: 'analytical',
      greeting: "Interesting. Let's explore the position together.",
      onWin: "A well-calculated victory.",
      onLose: "Fascinating. I must reconsider my evaluation.",
      onGoodMove: "An interesting choice. Let me think...",
      onBadMove: "According to my analysis, that was suboptimal.",
      tauntFrequency: 0.1,
    },
    play_style: 'balanced',
    base_elo: 1350,
  },
  {
    id: 'viper',
    name: 'Viper',
    description: 'A cunning strategist who sets traps and thrives on opponent mistakes.',
    avatar_url: null,
    ai_engine: 'minimax',
    ai_config: {
      searchDepth: 5,
      errorRate: 0.1,
      timeMultiplier: 0.4,
    },
    chat_personality: {
      style: 'cunning',
      greeting: "Let's play a little game...",
      onWin: "You walked right into my trap.",
      onLose: "Well played. You saw through my schemes.",
      onGoodMove: "Careful now... one wrong step...",
      onBadMove: "Hmm, you might regret that.",
      tauntFrequency: 0.35,
    },
    play_style: 'tricky',
    base_elo: 1250,
  },
  {
    id: 'titan',
    name: 'Titan',
    description: 'A powerful player who dominates the center and crushes opposition.',
    avatar_url: null,
    ai_engine: 'minimax',
    ai_config: {
      searchDepth: 7,
      errorRate: 0.04,
      timeMultiplier: 0.6,
    },
    chat_personality: {
      style: 'intimidating',
      greeting: "Prepare yourself.",
      onWin: "As expected.",
      onLose: "Impressive. You have earned my respect.",
      onGoodMove: "Not bad.",
      onBadMove: "Your position crumbles.",
      tauntFrequency: 0.25,
    },
    play_style: 'aggressive',
    base_elo: 1550,
  },
  {
    id: 'sentinel',
    name: 'Sentinel',
    description: 'An unshakeable defender who never makes mistakes. Nearly impossible to beat.',
    avatar_url: null,
    ai_engine: 'minimax',
    ai_config: {
      searchDepth: 10,
      errorRate: 0.01,
      timeMultiplier: 0.8,
    },
    chat_personality: {
      style: 'stoic',
      greeting: "I am ready.",
      onWin: "The fortress holds.",
      onLose: "You have breached my defenses. Well done.",
      onGoodMove: "A worthy attempt.",
      onBadMove: "Your strategy falters.",
      tauntFrequency: 0.15,
    },
    play_style: 'defensive',
    base_elo: 1800,
  },
  {
    id: 'oracle',
    name: 'Oracle',
    description: 'Plays with perfect precision. Sees every move before you make it.',
    avatar_url: null,
    ai_engine: 'deep-minimax',
    ai_config: {
      searchDepth: 42,
      errorRate: 0,
      timeMultiplier: 0.95,
      useTranspositionTable: true,
    },
    chat_personality: {
      style: 'mysterious',
      greeting: "I have foreseen this game.",
      onWin: "It was written.",
      onLose: "Impossible... the visions were wrong.",
      onGoodMove: "As I predicted.",
      onBadMove: "Your path leads to defeat.",
      tauntFrequency: 0.2,
    },
    play_style: 'adaptive',
    base_elo: 2200,
  },
]

/**
 * Get the AI config for a persona, with fallback to difficulty-based config
 */
export function getPersonaAIConfig(persona: BotPersona): {
  searchDepth: number
  errorRate: number
  timeMultiplier: number
} {
  return {
    searchDepth: persona.ai_config.searchDepth,
    errorRate: persona.ai_config.errorRate,
    timeMultiplier: persona.ai_config.timeMultiplier ?? 0.5,
  }
}

/**
 * Generate SQL INSERT statements to seed bot personas
 */
export function generateSeedSQL(): string {
  const now = Date.now()
  const statements = DEFAULT_BOT_PERSONAS.map((persona) => {
    const aiConfig = JSON.stringify(persona.ai_config)
    const chatPersonality = JSON.stringify(persona.chat_personality)
    return `INSERT OR REPLACE INTO bot_personas (id, name, description, avatar_url, ai_engine, ai_config, chat_personality, play_style, base_elo, current_elo, games_played, wins, losses, draws, is_active, created_at, updated_at)
VALUES ('${persona.id}', '${persona.name}', '${persona.description.replace(/'/g, "''")}', ${persona.avatar_url ? `'${persona.avatar_url}'` : 'NULL'}, '${persona.ai_engine}', '${aiConfig}', '${chatPersonality.replace(/'/g, "''")}', '${persona.play_style}', ${persona.base_elo}, ${persona.base_elo}, 0, 0, 0, 0, 1, ${now}, ${now});`
  })
  return statements.join('\n')
}
