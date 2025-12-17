import type { Locator, Page } from '@playwright/test'

/**
 * Page object for the Debug Game Render Page
 * Tests game rendering in isolation with comprehensive logging
 */
export class DebugRenderPage {
  readonly page: Page

  // Main containers
  readonly container: Locator
  readonly gameBoard: Locator
  readonly boardGrid: Locator

  // Game status indicators
  readonly currentPlayer: Locator
  readonly moveCount: Locator
  readonly winner: Locator
  readonly validMoves: Locator
  readonly renderCount: Locator
  readonly gameStatus: Locator

  // Controls
  readonly resetButton: Locator
  readonly undoButton: Locator
  readonly autoPlayButton: Locator
  readonly autoPlayInterval: Locator
  readonly clearLogsButton: Locator

  // Debug outputs
  readonly debugLog: Locator
  readonly moveHistory: Locator
  readonly boardJson: Locator

  constructor(page: Page) {
    this.page = page

    // Main containers
    this.container = page.locator('[data-testid="debug-game-render-page"]')
    this.gameBoard = page.locator('[data-testid="game-board"]')
    this.boardGrid = page.locator('[data-testid="board-grid"]')

    // Game status indicators
    this.currentPlayer = page.locator('[data-testid="current-player"]')
    this.moveCount = page.locator('[data-testid="move-count"]')
    this.winner = page.locator('[data-testid="winner"]')
    this.validMoves = page.locator('[data-testid="valid-moves"]')
    this.renderCount = page.locator('[data-testid="render-count"]')
    this.gameStatus = page.locator('[data-testid="game-status"]')

    // Controls
    this.resetButton = page.locator('[data-testid="reset-btn"]')
    this.undoButton = page.locator('[data-testid="undo-btn"]')
    this.autoPlayButton = page.locator('[data-testid="autoplay-btn"]')
    this.autoPlayInterval = page.locator('[data-testid="autoplay-interval"]')
    this.clearLogsButton = page.locator('[data-testid="clear-logs-btn"]')

    // Debug outputs
    this.debugLog = page.locator('[data-testid="debug-log"]')
    this.moveHistory = page.locator('[data-testid="move-history"]')
    this.boardJson = page.locator('[data-testid="board-json"]')
  }

  async goto() {
    await this.page.goto('/debug/game-render')
    await this.page.waitForLoadState('networkidle')
    await this.container.waitFor({ state: 'visible', timeout: 10000 })
  }

  /**
   * Get a column click target by index (0-6)
   */
  getColumn(column: number): Locator {
    return this.page.locator(`[data-testid="column-${column}"]`)
  }

  /**
   * Get a cell by row and column
   */
  getCell(row: number, col: number): Locator {
    return this.page.locator(`[data-testid="cell-${row}-${col}"]`)
  }

  /**
   * Get a piece by row and column (only exists if cell is occupied)
   */
  getPiece(row: number, col: number): Locator {
    return this.page.locator(`[data-testid="piece-${row}-${col}"]`)
  }

  /**
   * Click on a column to drop a piece
   */
  async dropPieceInColumn(column: number) {
    await this.getColumn(column).click()
    // Wait for state update
    await this.page.waitForTimeout(100)
  }

  /**
   * Play a sequence of moves (columns)
   */
  async playMoves(columns: number[]) {
    for (const col of columns) {
      await this.dropPieceInColumn(col)
      // Check if game is over
      const winnerText = await this.winner.textContent()
      if (winnerText && winnerText !== 'None') break
    }
  }

  /**
   * Get the current player number (1 or 2)
   */
  async getCurrentPlayer(): Promise<number> {
    const attr = await this.gameBoard.getAttribute('data-current-player')
    return parseInt(attr || '1')
  }

  /**
   * Get the winner from the game board data attribute
   */
  async getWinner(): Promise<string | null> {
    const attr = await this.gameBoard.getAttribute('data-winner')
    return attr === 'none' ? null : attr
  }

  /**
   * Get the current render count
   */
  async getRenderCount(): Promise<number> {
    const attr = await this.gameBoard.getAttribute('data-render-count')
    return parseInt(attr || '0')
  }

  /**
   * Get the page render count
   */
  async getPageRenderCount(): Promise<number> {
    const attr = await this.container.getAttribute('data-page-renders')
    return parseInt(attr || '0')
  }

  /**
   * Get the move count from the display
   */
  async getMoveCount(): Promise<number> {
    const text = await this.moveCount.textContent()
    return parseInt(text || '0')
  }

  /**
   * Check if a cell is empty
   */
  async isCellEmpty(row: number, col: number): Promise<boolean> {
    const cell = this.getCell(row, col)
    const cellValue = await cell.getAttribute('data-cell')
    return cellValue === 'empty'
  }

  /**
   * Get the player who occupies a cell (1, 2, or null if empty)
   */
  async getCellPlayer(row: number, col: number): Promise<number | null> {
    const cell = this.getCell(row, col)
    const cellValue = await cell.getAttribute('data-cell')
    if (cellValue === 'empty') return null
    return parseInt(cellValue || '0')
  }

  /**
   * Check if a cell is part of a winning combination
   */
  async isWinningCell(row: number, col: number): Promise<boolean> {
    const cell = this.getCell(row, col)
    const winning = await cell.getAttribute('data-winning')
    return winning === 'true'
  }

  /**
   * Get all winning cells
   */
  async getWinningCells(): Promise<Array<[number, number]>> {
    const winningCells: Array<[number, number]> = []
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 7; col++) {
        if (await this.isWinningCell(row, col)) {
          winningCells.push([row, col])
        }
      }
    }
    return winningCells
  }

  /**
   * Count total pieces on the board
   */
  async countPieces(): Promise<number> {
    const pieces = this.page.locator('[data-testid^="piece-"]')
    return await pieces.count()
  }

  /**
   * Count pieces for a specific player
   */
  async countPlayerPieces(player: 1 | 2): Promise<number> {
    const pieces = this.page.locator(`[data-testid^="piece-"][data-player="${player}"]`)
    return await pieces.count()
  }

  /**
   * Reset the game
   */
  async reset() {
    await this.resetButton.click()
    await this.page.waitForTimeout(100)
  }

  /**
   * Undo the last move
   */
  async undo() {
    await this.undoButton.click()
    await this.page.waitForTimeout(100)
  }

  /**
   * Start auto-play mode
   */
  async startAutoPlay() {
    const buttonText = await this.autoPlayButton.textContent()
    if (buttonText?.includes('Start')) {
      await this.autoPlayButton.click()
    }
  }

  /**
   * Stop auto-play mode
   */
  async stopAutoPlay() {
    const buttonText = await this.autoPlayButton.textContent()
    if (buttonText?.includes('Stop')) {
      await this.autoPlayButton.click()
    }
  }

  /**
   * Set the auto-play interval (range input, min 100, max 2000)
   */
  async setAutoPlayInterval(ms: number) {
    // Range inputs need special handling
    await this.page.evaluate((value) => {
      const slider = document.querySelector('[data-testid="autoplay-interval"]') as HTMLInputElement
      if (slider) {
        slider.value = String(value)
        slider.dispatchEvent(new Event('change', { bubbles: true }))
      }
    }, ms)
  }

  /**
   * Clear the debug logs
   */
  async clearLogs() {
    await this.clearLogsButton.click()
  }

  /**
   * Get the debug log content
   */
  async getDebugLogContent(): Promise<string> {
    return (await this.debugLog.inputValue()) || ''
  }

  /**
   * Get the move history as an array
   */
  async getMoveHistory(): Promise<number[]> {
    const text = await this.moveHistory.textContent()
    if (!text || text === 'No moves yet') return []
    return text.split(',').map((s) => parseInt(s.trim()))
  }

  /**
   * Get the board state from JSON display
   */
  async getBoardState(): Promise<{
    board: (number | null)[][]
    currentPlayer: number
    winner: string | null
    moveHistory: number[]
  }> {
    const json = await this.boardJson.textContent()
    return JSON.parse(json || '{}')
  }

  /**
   * Wait for a specific number of renders
   */
  async waitForRenderCount(count: number, timeout: number = 5000) {
    const startTime = Date.now()
    while (Date.now() - startTime < timeout) {
      const currentCount = await this.getRenderCount()
      if (currentCount >= count) return
      await this.page.waitForTimeout(50)
    }
    throw new Error(`Timeout waiting for render count ${count}`)
  }

  /**
   * Wait for game to end (winner or draw)
   */
  async waitForGameEnd(timeout: number = 30000) {
    const startTime = Date.now()
    while (Date.now() - startTime < timeout) {
      const winner = await this.getWinner()
      if (winner !== null) return winner
      await this.page.waitForTimeout(100)
    }
    throw new Error('Timeout waiting for game to end')
  }

  /**
   * Check if the board is empty
   */
  async isBoardEmpty(): Promise<boolean> {
    const pieceCount = await this.countPieces()
    return pieceCount === 0
  }

  /**
   * Verify board dimensions (6 rows x 7 columns)
   */
  async verifyBoardDimensions(): Promise<boolean> {
    // Check all cells exist
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 7; col++) {
        const cell = this.getCell(row, col)
        if (!(await cell.isVisible())) {
          return false
        }
      }
    }
    return true
  }
}
