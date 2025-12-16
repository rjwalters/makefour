/**
 * ELO Rating System calculations for MakeFour
 *
 * Uses standard ELO formula with variable K-factor:
 * - K=32 for new players (<30 games) - ratings change more quickly
 * - K=16 for established players - ratings are more stable
 */

/** Default starting rating for new players */
export const DEFAULT_RATING = 1200

/** K-factor for new players (fewer than 30 games) */
const K_FACTOR_NEW = 32

/** K-factor for established players (30+ games) */
const K_FACTOR_ESTABLISHED = 16

/** Threshold for considering a player "established" */
const ESTABLISHED_GAMES_THRESHOLD = 30

/**
 * Game outcome from the player's perspective
 */
export type GameOutcome = 'win' | 'loss' | 'draw'

/**
 * Result of an ELO calculation
 */
export interface EloResult {
  newRating: number
  ratingChange: number
}

/**
 * Gets the appropriate K-factor based on games played
 */
function getKFactor(gamesPlayed: number): number {
  return gamesPlayed < ESTABLISHED_GAMES_THRESHOLD ? K_FACTOR_NEW : K_FACTOR_ESTABLISHED
}

/**
 * Calculates the expected score (probability of winning) using the ELO formula.
 *
 * E = 1 / (1 + 10^((opponentRating - playerRating) / 400))
 *
 * @param playerRating - The player's current rating
 * @param opponentRating - The opponent's rating
 * @returns Expected score between 0 and 1
 */
function calculateExpectedScore(playerRating: number, opponentRating: number): number {
  return 1 / (1 + 10 ** ((opponentRating - playerRating) / 400))
}

/**
 * Converts game outcome to a score value for ELO calculation.
 *
 * @param outcome - The game outcome from player's perspective
 * @returns 1 for win, 0.5 for draw, 0 for loss
 */
function outcomeToScore(outcome: GameOutcome): number {
  switch (outcome) {
    case 'win':
      return 1
    case 'draw':
      return 0.5
    case 'loss':
      return 0
  }
}

/**
 * Calculates the new ELO rating after a game.
 *
 * newRating = oldRating + K * (actualScore - expectedScore)
 *
 * @param playerRating - The player's current rating
 * @param opponentRating - The opponent's rating (for AI, use AI_RATING constant)
 * @param outcome - The game outcome from the player's perspective
 * @param gamesPlayed - Number of games the player has played (for K-factor)
 * @returns The new rating and the rating change
 */
export function calculateNewRating(
  playerRating: number,
  opponentRating: number,
  outcome: GameOutcome,
  gamesPlayed: number
): EloResult {
  const kFactor = getKFactor(gamesPlayed)
  const expectedScore = calculateExpectedScore(playerRating, opponentRating)
  const actualScore = outcomeToScore(outcome)

  const ratingChange = Math.round(kFactor * (actualScore - expectedScore))
  const newRating = playerRating + ratingChange

  // Ensure rating doesn't go below 100 (floor)
  const finalRating = Math.max(100, newRating)

  return {
    newRating: finalRating,
    ratingChange: finalRating - playerRating,
  }
}

/**
 * AI difficulty ratings for ELO calculations.
 * When playing against AI, use these ratings as the "opponent rating".
 */
export const AI_RATINGS = {
  easy: 800,
  medium: 1200,
  hard: 1600,
  expert: 2000,
} as const

export type AIDifficulty = keyof typeof AI_RATINGS

/**
 * Gets the AI rating for a given difficulty level.
 * Defaults to medium difficulty (1200) if not specified.
 */
export function getAIRating(difficulty?: AIDifficulty): number {
  if (!difficulty || !(difficulty in AI_RATINGS)) {
    return AI_RATINGS.medium
  }
  return AI_RATINGS[difficulty]
}
