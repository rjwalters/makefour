import type { Locator, Page } from '@playwright/test'
import { timeouts } from '../fixtures/test-data'

/**
 * Page object for the Play page (game board)
 */
export class PlayPage {
  readonly page: Page
  readonly pageTitle: Locator
  readonly gameBoard: Locator
  readonly newGameButton: Locator
  readonly saveGameButton: Locator
  readonly currentPlayerIndicator: Locator
  readonly winnerIndicator: Locator
  readonly logoutButton: Locator
  readonly dashboardLink: Locator
  readonly mobileMenuButton: Locator

  // Game mode selectors
  readonly aiModeButton: Locator
  readonly hotseatModeButton: Locator
  readonly onlineModeButton: Locator
  readonly startGameButton: Locator
  readonly difficultySelector: Locator

  constructor(page: Page) {
    this.page = page
    this.pageTitle = page.locator('text=MakeFour').first()
    this.gameBoard = page.locator('.bg-blue-600, .bg-blue-800').first()
    this.newGameButton = page.locator('button:has-text("New Game")')
    this.saveGameButton = page.locator('button:has-text("Save")')
    this.currentPlayerIndicator = page.locator('text=/Player [12]/').first()
    this.winnerIndicator = page.locator('text=/wins|Draw|Winner/')
    this.logoutButton = page.locator('button:has-text("Logout")')
    this.dashboardLink = page.locator('a[href="/dashboard"]').first()
    this.mobileMenuButton = page.locator('button[aria-label="Toggle menu"]')

    // Game mode buttons
    this.aiModeButton = page.locator('button:has-text("vs AI"), button:has-text("AI")')
    this.hotseatModeButton = page.locator('button:has-text("Hotseat"), button:has-text("Local")')
    this.onlineModeButton = page.locator('button:has-text("Online")')
    this.startGameButton = page.locator(
      'button:has-text("Start Game"), button:has-text("Start"), button:has-text("Play")'
    )
    this.difficultySelector = page.locator('select, [role="combobox"]')
  }

  async goto() {
    await this.page.goto('/play')
    await this.page.waitForLoadState('networkidle')
  }

  /**
   * Get a column button by column index (0-6)
   */
  getColumnButton(column: number): Locator {
    return this.page.locator(`button[aria-label="Drop piece in column ${column + 1}"]`)
  }

  /**
   * Get a cell by row and column index
   */
  getCell(row: number, column: number): Locator {
    return this.page.locator(`button[aria-label*="column ${column + 1}"]`).nth(row + 1) // +1 to skip header buttons
  }

  /**
   * Click on a column to drop a piece
   */
  async dropPieceInColumn(column: number) {
    const columnButton = this.getColumnButton(column)
    await columnButton.click()
    // Wait for the animation/state update
    await this.page.waitForTimeout(100)
  }

  /**
   * Start a new hotseat game (local 2-player)
   */
  async startHotseatGame() {
    // Look for hotseat/local mode
    const hotseatBtn = this.page.locator(
      'button:has-text("Hotseat"), button:has-text("Local 2P"), label:has-text("Hotseat")'
    )
    if (await hotseatBtn.isVisible()) {
      await hotseatBtn.click()
    }

    // Start the game
    const startBtn = this.page.locator(
      'button:has-text("Start Game"), button:has-text("Start"), button:has-text("Play Now")'
    )
    if (await startBtn.isVisible()) {
      await startBtn.click()
    }

    await this.page.waitForTimeout(500)
  }

  /**
   * Start a new game against AI
   */
  async startAIGame(difficulty: string = 'easy') {
    const aiBtn = this.page.locator('button:has-text("vs AI"), button:has-text("AI"), label:has-text("AI")')
    if (await aiBtn.isVisible()) {
      await aiBtn.click()
    }

    // Select difficulty if available
    const difficultySelect = this.page.locator(`button:has-text("${difficulty}"), option:has-text("${difficulty}")`)
    if (await difficultySelect.isVisible()) {
      await difficultySelect.click()
    }

    const startBtn = this.page.locator(
      'button:has-text("Start Game"), button:has-text("Start"), button:has-text("Play Now")'
    )
    if (await startBtn.isVisible()) {
      await startBtn.click()
    }

    await this.page.waitForTimeout(500)
  }

  /**
   * Play a sequence of moves (columns)
   */
  async playMoves(columns: number[]) {
    for (const col of columns) {
      await this.dropPieceInColumn(col)
      await this.page.waitForTimeout(200) // Wait between moves
    }
  }

  /**
   * Check if the game is over (winner or draw)
   */
  async isGameOver(): Promise<boolean> {
    const winText = this.page.locator('text=/wins|Draw|Winner|Game Over/')
    return await winText.isVisible()
  }

  /**
   * Get the winner (1, 2, or 'draw')
   */
  async getWinner(): Promise<string | null> {
    const player1Wins = await this.page.locator('text=/Player 1 wins|Red wins/i').isVisible()
    const player2Wins = await this.page.locator('text=/Player 2 wins|Yellow wins/i').isVisible()
    const isDraw = await this.page.locator('text=/Draw|Tie/i').isVisible()

    if (player1Wins) return '1'
    if (player2Wins) return '2'
    if (isDraw) return 'draw'
    return null
  }

  /**
   * Click new game button
   */
  async clickNewGame() {
    await this.newGameButton.click()
    await this.page.waitForTimeout(500)
  }

  /**
   * Save the current game
   */
  async saveGame() {
    await this.saveGameButton.click()
    await this.page.waitForTimeout(1000)
  }

  /**
   * Navigate to dashboard
   */
  async navigateToDashboard() {
    const isMobileMenuVisible = await this.mobileMenuButton.isVisible()
    if (isMobileMenuVisible) {
      await this.mobileMenuButton.click()
    }

    // Try dashboard link first, then logo
    const dashLink = this.page.locator('a[href="/dashboard"]').first()
    if (await dashLink.isVisible()) {
      await dashLink.click()
    } else {
      await this.pageTitle.click()
    }

    await this.page.waitForURL('/dashboard', { timeout: timeouts.navigation })
  }

  /**
   * Check if board has any pieces
   */
  async hasPieces(): Promise<boolean> {
    // Look for colored pieces (red or yellow backgrounds)
    const pieces = this.page.locator('.bg-red-500, .bg-yellow-400')
    const count = await pieces.count()
    return count > 0
  }

  /**
   * Wait for AI to make a move
   */
  async waitForAIMove(timeout: number = 5000) {
    await this.page.waitForTimeout(timeout)
  }
}
