/**
 * Test data and fixtures for E2E tests
 */

export interface TestUser {
  email: string
  password: string
}

/**
 * Generate a unique test user for each test run
 * Uses timestamp to ensure uniqueness
 */
export function generateTestUser(): TestUser {
  const timestamp = Date.now()
  return {
    email: `test-${timestamp}@e2e-test.makefour.local`,
    password: `TestPass123!${timestamp}`,
  }
}

/**
 * Pre-defined test users for specific test scenarios
 */
export const testUsers = {
  standard: generateTestUser,
  // For tests that need a consistent user across runs
  persistent: {
    email: 'e2e-persistent@test.makefour.local',
    password: 'PersistentTestPass123!',
  },
}

/**
 * Game board configurations for testing
 */
export const testBoards = {
  // Empty board
  empty: Array(6)
    .fill(null)
    .map(() => Array(7).fill(null)),

  // Board with a winning position for player 1 (horizontal)
  horizontalWin: [
    [null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null],
    [2, 2, 2, null, null, null, null],
    [1, 1, 1, null, 2, null, null], // Player 1 needs column 3 to win
  ],

  // Board with a winning position for player 1 (vertical)
  verticalWin: [
    [null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null],
    [null, null, null, null, null, null, null],
    [1, 2, null, null, null, null, null],
    [1, 2, null, null, null, null, null],
    [1, 2, null, null, null, null, null], // Player 1 needs column 0 to win
  ],
}

/**
 * Test timeouts
 */
export const timeouts = {
  navigation: 10000,
  animation: 1000,
  network: 30000,
}

/**
 * Test selectors for commonly used elements
 */
export const selectors = {
  // Login page
  loginForm: {
    email: 'input#email',
    password: 'input#password',
    submitButton: 'button[type="submit"]',
    toggleMode: 'button:has-text("New user?")',
    googleButton: 'button:has-text("Continue with Google")',
    errorMessage: '.text-destructive',
  },

  // Navigation
  nav: {
    dashboard: '[href="/dashboard"]',
    play: '[href="/play"]',
    games: '[href="/games"]',
    leaderboard: '[href="/leaderboard"]',
    profile: '[href="/profile"]',
    logout: 'button:has-text("Logout")',
  },

  // Game board
  board: {
    container: '[data-testid="game-board"]',
    column: '[data-testid^="column-"]',
    cell: '[data-testid^="cell-"]',
    currentPlayer: '[data-testid="current-player"]',
    winner: '[data-testid="winner"]',
    newGameButton: 'button:has-text("New Game")',
    saveGameButton: 'button:has-text("Save")',
  },

  // Game history
  history: {
    gameList: '[data-testid="game-list"]',
    gameItem: '[data-testid^="game-"]',
    replayButton: 'button:has-text("Replay")',
  },

  // Replay controls
  replay: {
    container: '[data-testid="replay-container"]',
    playButton: 'button:has-text("Play")',
    pauseButton: 'button:has-text("Pause")',
    prevButton: 'button[aria-label="Previous move"]',
    nextButton: 'button[aria-label="Next move"]',
    moveSlider: '[data-testid="move-slider"]',
    moveCounter: '[data-testid="move-counter"]',
  },
}
