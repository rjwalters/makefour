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
    this.gameBoard = page.locator('.bg-blue-600, .dark\\:bg-blue-800').first()
    this.newGameButton = page.locator('button:has-text("New Game"), button:has-text("Change Settings")')
    this.saveGameButton = page.locator('button:has-text("Save")')
    this.currentPlayerIndicator = page.locator('text=/Player [12]/').first()
    this.winnerIndicator = page.locator('text=/wins|Draw|Winner|You win|AI wins/')
    this.logoutButton = page.locator('button:has-text("Logout")')
    this.dashboardLink = page.locator('a[href="/dashboard"]').first()
    this.mobileMenuButton = page.locator('button[aria-label="Toggle menu"]')

    // Game mode buttons - Training Mode is the default now
    this.aiModeButton = page.locator('button:has-text("vs AI"), button:has-text("AI")')
    this.hotseatModeButton = page.locator('button:has-text("Hotseat"), button:has-text("Local")')
    this.onlineModeButton = page.locator('button:has-text("Online")')
    this.startGameButton = page.locator(
      'button:has-text("Start Training"), button:has-text("Start Game"), button:has-text("Start"), button:has-text("Play")'
    )
    this.difficultySelector = page.locator('select, [role="combobox"]')
  }

  async goto() {
    await this.page.goto('/play?mode=training')
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
   * Start a new training game (vs AI with coaching)
   * Note: This was previously hotseat, but the UI now defaults to Training Mode
   */
  async startHotseatGame() {
    // Training Mode is the default - just click the start button
    const startBtn = this.page.locator(
      'button:has-text("Start Training"), button:has-text("Start Game"), button:has-text("Start"), button:has-text("Play Now")'
    )
    if (await startBtn.isVisible()) {
      await startBtn.click()
    }

    // Wait for the game to start and board to appear
    await this.page.waitForTimeout(500)
    await this.gameBoard.waitFor({ state: 'visible', timeout: 5000 })
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
   * In Training Mode (AI), waits for AI to move after each player move
   */
  async playMoves(columns: number[]) {
    for (const col of columns) {
      await this.dropPieceInColumn(col)
      // Wait for move to register and check if game is over
      await this.page.waitForTimeout(200)

      // Check if game is over after this move
      const isOver = await this.isGameOver()
      if (isOver) break

      // In AI mode, wait for AI to make its move (look for "Your turn" indicator)
      try {
        await this.page.waitForSelector('text=/Your turn|wins|Draw/', { timeout: 5000 })
      } catch {
        // If timeout, the turn indicator might be different or game is over
      }
      await this.page.waitForTimeout(200)
    }
  }

  /**
   * Check if the game is over (winner or draw)
   */
  async isGameOver(): Promise<boolean> {
    const winText = this.page.locator('text=/wins|Draw|Winner|Game Over|You win|AI wins/')
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
