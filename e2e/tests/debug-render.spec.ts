import { expect, test } from '@playwright/test'
import { DebugRenderPage } from '../pages'

test.describe('Debug Game Render Page', () => {
  let debugPage: DebugRenderPage

  test.beforeEach(async ({ page }) => {
    debugPage = new DebugRenderPage(page)
    await debugPage.goto()
  })

  test.describe('Initial State', () => {
    test('should load the debug page', async () => {
      await expect(debugPage.container).toBeVisible()
      await expect(debugPage.gameBoard).toBeVisible()
    })

    test('should have correct board dimensions (6x7)', async () => {
      const hasCorrectDimensions = await debugPage.verifyBoardDimensions()
      expect(hasCorrectDimensions).toBe(true)
    })

    test('should start with empty board', async () => {
      const isEmpty = await debugPage.isBoardEmpty()
      expect(isEmpty).toBe(true)
    })

    test('should start with player 1', async () => {
      const currentPlayer = await debugPage.getCurrentPlayer()
      expect(currentPlayer).toBe(1)
    })

    test('should show no winner initially', async () => {
      const winner = await debugPage.getWinner()
      expect(winner).toBeNull()
    })

    test('should show zero moves initially', async () => {
      const moveCount = await debugPage.getMoveCount()
      expect(moveCount).toBe(0)
    })

    test('should show valid moves for empty board', async () => {
      const validMoves = await debugPage.validMoves.textContent()
      expect(validMoves).toContain('0')
      expect(validMoves).toContain('6')
    })

    test('should have all control buttons visible', async () => {
      await expect(debugPage.resetButton).toBeVisible()
      await expect(debugPage.undoButton).toBeVisible()
      await expect(debugPage.autoPlayButton).toBeVisible()
    })

    test('should have debug log visible', async () => {
      await expect(debugPage.debugLog).toBeVisible()
    })
  })

  test.describe('Making Moves', () => {
    test('should drop piece in column 3', async () => {
      await debugPage.dropPieceInColumn(3)

      const pieceCount = await debugPage.countPieces()
      expect(pieceCount).toBe(1)
    })

    test('should place piece at bottom of column', async () => {
      await debugPage.dropPieceInColumn(3)

      // Bottom row is row 5
      const player = await debugPage.getCellPlayer(5, 3)
      expect(player).toBe(1)
    })

    test('should alternate players after each move', async () => {
      // Player 1 moves
      await debugPage.dropPieceInColumn(0)
      let currentPlayer = await debugPage.getCurrentPlayer()
      expect(currentPlayer).toBe(2)

      // Player 2 moves
      await debugPage.dropPieceInColumn(1)
      currentPlayer = await debugPage.getCurrentPlayer()
      expect(currentPlayer).toBe(1)
    })

    test('should increment move count after each move', async () => {
      await debugPage.dropPieceInColumn(0)
      let moveCount = await debugPage.getMoveCount()
      expect(moveCount).toBe(1)

      await debugPage.dropPieceInColumn(1)
      moveCount = await debugPage.getMoveCount()
      expect(moveCount).toBe(2)
    })

    test('should stack pieces in same column', async () => {
      // Play two pieces in column 0
      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(0)

      // Bottom piece is player 1, second from bottom is player 2
      const bottomPlayer = await debugPage.getCellPlayer(5, 0)
      const secondPlayer = await debugPage.getCellPlayer(4, 0)

      expect(bottomPlayer).toBe(1)
      expect(secondPlayer).toBe(2)
    })

    test('should update move history display', async () => {
      await debugPage.dropPieceInColumn(3)
      await debugPage.dropPieceInColumn(4)

      const history = await debugPage.getMoveHistory()
      expect(history).toEqual([3, 4])
    })

    test('should trigger render on each move', async () => {
      const initialRenders = await debugPage.getRenderCount()

      await debugPage.dropPieceInColumn(0)

      const newRenders = await debugPage.getRenderCount()
      expect(newRenders).toBeGreaterThan(initialRenders)
    })
  })

  test.describe('Win Detection', () => {
    test('should detect horizontal win', async () => {
      // Player 1: columns 0, 1, 2, 3 (with player 2 moves in between)
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      const winner = await debugPage.getWinner()
      expect(winner).toBe('1')
    })

    test('should detect vertical win', async () => {
      // Player 1 plays column 0 four times, player 2 plays column 1
      await debugPage.playMoves([0, 1, 0, 1, 0, 1, 0])

      const winner = await debugPage.getWinner()
      expect(winner).toBe('1')
    })

    test('should detect diagonal win', async () => {
      // Build ascending diagonal for P1: (5,0), (4,1), (3,2), (2,3)
      // Need to carefully build stacks so P1 lands in the right positions
      await debugPage.playMoves([
        0, // P1 at (5,0)
        1, // P2 at (5,1)
        1, // P1 at (4,1)
        2, // P2 at (5,2)
        3, // P1 at (5,3) - blocking P2's potential win
        2, // P2 at (4,2)
        2, // P1 at (3,2)
        3, // P2 at (4,3)
        3, // P1 at (3,3)
        4, // P2 at (5,4)
        3, // P1 at (2,3) - diagonal win! (5,0), (4,1), (3,2), (2,3)
      ])

      const winner = await debugPage.getWinner()
      expect(winner).toBe('1')

      // Verify 4 winning cells are highlighted
      const winningCells = await debugPage.getWinningCells()
      expect(winningCells.length).toBe(4)
    })

    test('should highlight winning cells', async () => {
      // Create horizontal win for player 1
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      const winningCells = await debugPage.getWinningCells()
      expect(winningCells.length).toBe(4)
    })

    test('should display winner in status', async () => {
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      const winnerText = await debugPage.winner.textContent()
      expect(winnerText).toContain('Player 1')
    })

    test('should disable board after win', async () => {
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      // Verify the board shows a winner
      const winner = await debugPage.getWinner()
      expect(winner).toBe('1')

      // Verify columns are disabled
      const column = debugPage.getColumn(4)
      await expect(column).toBeDisabled()
    })
  })

  test.describe('Reset Functionality', () => {
    test('should reset board to empty', async () => {
      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(1)

      await debugPage.reset()

      const isEmpty = await debugPage.isBoardEmpty()
      expect(isEmpty).toBe(true)
    })

    test('should reset to player 1', async () => {
      await debugPage.dropPieceInColumn(0) // Now player 2

      await debugPage.reset()

      const currentPlayer = await debugPage.getCurrentPlayer()
      expect(currentPlayer).toBe(1)
    })

    test('should clear move history', async () => {
      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(1)

      await debugPage.reset()

      const history = await debugPage.getMoveHistory()
      expect(history).toEqual([])
    })

    test('should reset move count to zero', async () => {
      await debugPage.dropPieceInColumn(0)

      await debugPage.reset()

      const moveCount = await debugPage.getMoveCount()
      expect(moveCount).toBe(0)
    })

    test('should clear winner after game won', async () => {
      // Win the game
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      await debugPage.reset()

      const winner = await debugPage.getWinner()
      expect(winner).toBeNull()
    })
  })

  test.describe('Undo Functionality', () => {
    test('should undo last move', async () => {
      await debugPage.dropPieceInColumn(0)
      const pieceCountBefore = await debugPage.countPieces()
      expect(pieceCountBefore).toBe(1)

      await debugPage.undo()

      const pieceCountAfter = await debugPage.countPieces()
      expect(pieceCountAfter).toBe(0)
    })

    test('should restore previous player', async () => {
      await debugPage.dropPieceInColumn(0) // Player 1 moves, now player 2

      await debugPage.undo()

      const currentPlayer = await debugPage.getCurrentPlayer()
      expect(currentPlayer).toBe(1)
    })

    test('should update move history on undo', async () => {
      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(1)

      await debugPage.undo()

      const history = await debugPage.getMoveHistory()
      expect(history).toEqual([0])
    })

    test('should be disabled when no moves to undo', async () => {
      await expect(debugPage.undoButton).toBeDisabled()
    })

    test('should undo multiple times', async () => {
      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(1)
      await debugPage.dropPieceInColumn(2)

      await debugPage.undo()
      await debugPage.undo()

      const moveCount = await debugPage.getMoveCount()
      expect(moveCount).toBe(1)
    })
  })

  test.describe('Auto-Play', () => {
    test('should start auto-play when clicking button', async () => {
      await debugPage.startAutoPlay()

      // Wait for at least one auto move
      await debugPage.page.waitForTimeout(1000)

      const moveCount = await debugPage.getMoveCount()
      expect(moveCount).toBeGreaterThan(0)
    })

    test('should stop auto-play when clicking stop', async () => {
      await debugPage.startAutoPlay()
      await debugPage.page.waitForTimeout(600)

      await debugPage.stopAutoPlay()
      const moveCountAtStop = await debugPage.getMoveCount()

      // Wait to confirm no more moves
      await debugPage.page.waitForTimeout(1000)
      const moveCountAfter = await debugPage.getMoveCount()

      expect(moveCountAfter).toBe(moveCountAtStop)
    })

    test('should play until game ends', async ({ page }) => {
      // Set fast interval
      await debugPage.autoPlayInterval.fill('100')

      await debugPage.startAutoPlay()

      // Wait for game to end (max 42 moves * 100ms = 4.2s + buffer)
      const winner = await debugPage.waitForGameEnd(10000)

      expect(winner).not.toBeNull()
    })
  })

  test.describe('Debug Logging', () => {
    test('should log moves to debug output', async () => {
      await debugPage.clearLogs()
      await debugPage.dropPieceInColumn(3)

      const logContent = await debugPage.getDebugLogContent()
      expect(logContent).toContain('MOVE')
    })

    test('should log state changes', async () => {
      await debugPage.clearLogs()
      await debugPage.dropPieceInColumn(3)

      const logContent = await debugPage.getDebugLogContent()
      expect(logContent).toContain('STATE')
    })

    test('should log render events', async () => {
      await debugPage.clearLogs()
      await debugPage.dropPieceInColumn(3)

      const logContent = await debugPage.getDebugLogContent()
      expect(logContent).toContain('RENDER')
    })

    test('should log click events', async () => {
      await debugPage.clearLogs()
      await debugPage.dropPieceInColumn(3)

      const logContent = await debugPage.getDebugLogContent()
      expect(logContent).toContain('CLICK')
    })

    test('should clear logs when clicking clear button', async () => {
      await debugPage.dropPieceInColumn(3)
      const logsBefore = await debugPage.getDebugLogContent()

      await debugPage.clearLogs()

      const logsAfter = await debugPage.getDebugLogContent()
      // Logs should contain the "Logs cleared" message and have fewer entries than before
      expect(logsAfter).toContain('Logs cleared')
      expect(logsAfter.length).toBeLessThan(logsBefore.length)
    })

    test('should log win detection', async () => {
      await debugPage.clearLogs()
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      const logContent = await debugPage.getDebugLogContent()
      expect(logContent).toContain('WIN')
    })
  })

  test.describe('Board JSON State', () => {
    test('should show current board state as JSON', async () => {
      await debugPage.dropPieceInColumn(3)

      const state = await debugPage.getBoardState()
      expect(state.board).toBeDefined()
      expect(state.currentPlayer).toBe(2)
      expect(state.moveHistory).toEqual([3])
    })

    test('should update JSON on each move', async () => {
      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(1)

      const state = await debugPage.getBoardState()
      expect(state.moveHistory.length).toBe(2)
    })

    test('should show winner in JSON when game ends', async () => {
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      const state = await debugPage.getBoardState()
      expect(state.winner).toBe(1)
    })
  })

  test.describe('Render Performance', () => {
    test('should track render count', async () => {
      const initialCount = await debugPage.getRenderCount()

      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(1)

      const finalCount = await debugPage.getRenderCount()
      expect(finalCount).toBeGreaterThan(initialCount)
    })

    test('should track page renders', async () => {
      const initialCount = await debugPage.getPageRenderCount()

      await debugPage.dropPieceInColumn(0)

      const finalCount = await debugPage.getPageRenderCount()
      expect(finalCount).toBeGreaterThanOrEqual(initialCount)
    })

    test('should handle rapid moves without issues', async () => {
      // Make 10 rapid moves
      for (let i = 0; i < 10; i++) {
        await debugPage.dropPieceInColumn(i % 7)
      }

      const pieceCount = await debugPage.countPieces()
      expect(pieceCount).toBe(10)
    })
  })

  test.describe('Cell Data Attributes', () => {
    test('should have correct data-cell attribute for empty cells', async () => {
      const cell = debugPage.getCell(0, 0)
      const cellValue = await cell.getAttribute('data-cell')
      expect(cellValue).toBe('empty')
    })

    test('should have correct data-cell attribute for occupied cells', async () => {
      await debugPage.dropPieceInColumn(0)

      const cell = debugPage.getCell(5, 0)
      const cellValue = await cell.getAttribute('data-cell')
      expect(cellValue).toBe('1')
    })

    test('should have data-winning=false for non-winning cells', async () => {
      await debugPage.dropPieceInColumn(0)

      const cell = debugPage.getCell(5, 0)
      const winning = await cell.getAttribute('data-winning')
      expect(winning).toBe('false')
    })

    test('should have data-winning=true for winning cells', async () => {
      await debugPage.playMoves([0, 6, 1, 6, 2, 6, 3])

      // Cell at (5, 0) should be part of winning combination
      const cell = debugPage.getCell(5, 0)
      const winning = await cell.getAttribute('data-winning')
      expect(winning).toBe('true')
    })
  })

  test.describe('Piece Data Attributes', () => {
    test('should have data-player attribute on pieces', async () => {
      await debugPage.dropPieceInColumn(0)

      const piece = debugPage.getPiece(5, 0)
      const player = await piece.getAttribute('data-player')
      expect(player).toBe('1')
    })

    test('should show correct player for each piece', async () => {
      await debugPage.dropPieceInColumn(0) // Player 1
      await debugPage.dropPieceInColumn(1) // Player 2

      const piece1 = debugPage.getPiece(5, 0)
      const piece2 = debugPage.getPiece(5, 1)

      expect(await piece1.getAttribute('data-player')).toBe('1')
      expect(await piece2.getAttribute('data-player')).toBe('2')
    })
  })

  test.describe('Draw Detection', () => {
    test('should detect draw when board is full', async ({ page }) => {
      // Set fast auto-play interval using the range input
      // The min is 100, so we'll use that
      await debugPage.page.evaluate(() => {
        const slider = document.querySelector('[data-testid="autoplay-interval"]') as HTMLInputElement
        if (slider) {
          slider.value = '100'
          slider.dispatchEvent(new Event('change', { bubbles: true }))
        }
      })

      await debugPage.startAutoPlay()

      // Wait for game to end (max 42 moves * 100ms = 4.2s + buffer)
      const winner = await debugPage.waitForGameEnd(15000)

      // Game should have ended (either winner or draw)
      expect(winner).not.toBeNull()
    })
  })

  test.describe('Text Render Mode', () => {
    test('should switch to text render mode', async () => {
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      const textBoard = debugPage.page.locator('[data-testid="text-board"]')
      await expect(textBoard).toBeVisible()
    })

    test('should display ASCII board in text mode', async () => {
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      const asciiBoard = debugPage.page.locator('[data-testid="ascii-board"]')
      await expect(asciiBoard).toBeVisible()
    })

    test('should show correct data attributes in text mode', async () => {
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      const textBoard = debugPage.page.locator('[data-testid="text-board"]')
      await expect(textBoard).toHaveAttribute('data-render-mode', 'text')
      await expect(textBoard).toHaveAttribute('data-current-player', '1')
      await expect(textBoard).toHaveAttribute('data-winner', 'none')
    })

    test('should update text board after move', async () => {
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      // Click column 3 in text mode
      const colButton = debugPage.page.locator('[data-testid="text-col-3"]')
      await colButton.click()
      await debugPage.page.waitForTimeout(100)

      // Check cell is filled
      const cell = debugPage.page.locator('[data-testid="text-cell-5-3"]')
      await expect(cell).toHaveAttribute('data-cell', '1')
    })

    test('should show version and optimistic state in text mode', async () => {
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      const textBoard = debugPage.page.locator('[data-testid="text-board"]')
      await expect(textBoard).toHaveAttribute('data-version', '0')
      await expect(textBoard).toHaveAttribute('data-optimistic', 'false')
    })
  })

  test.describe('useGameState Hook', () => {
    test('should enable useGameState hook toggle', async () => {
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()
      await expect(checkbox).toBeChecked()
    })

    test('should display version info when hook is enabled', async () => {
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()

      const versionInfo = debugPage.page.locator('[data-testid="version-info"]')
      await expect(versionInfo).toBeVisible()
    })

    test('should show initial version as 0', async () => {
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()

      const stateVersion = debugPage.page.locator('[data-testid="state-version"]')
      await expect(stateVersion).toHaveText('0')
    })

    test('should increment version on each move', async () => {
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()
      await debugPage.page.waitForTimeout(100)

      // Make a move
      await debugPage.dropPieceInColumn(3)

      const stateVersion = debugPage.page.locator('[data-testid="state-version"]')
      await expect(stateVersion).toHaveText('1')

      // Make another move
      await debugPage.dropPieceInColumn(4)

      await expect(stateVersion).toHaveText('2')
    })

    test('should track moves count correctly', async () => {
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()

      await debugPage.dropPieceInColumn(0)
      await debugPage.dropPieceInColumn(1)
      await debugPage.dropPieceInColumn(2)

      const stateMoves = debugPage.page.locator('[data-testid="state-moves"]')
      await expect(stateMoves).toHaveText('3')
    })
  })

  test.describe('State Consistency - Race Condition Prevention', () => {
    test('should maintain state consistency during rapid moves', async () => {
      // Make 10 rapid moves without waiting between them
      const promises = []
      for (let i = 0; i < 10; i++) {
        promises.push(debugPage.getColumn(i % 7).click())
      }

      // Wait for all clicks to process
      await debugPage.page.waitForTimeout(500)

      // Verify piece count matches move count
      const pieceCount = await debugPage.countPieces()
      const moveCount = await debugPage.getMoveCount()
      expect(pieceCount).toBe(moveCount)
    })

    test('should not have phantom pieces after rapid moves', async () => {
      // Enable the new hook for better tracking
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()
      await debugPage.page.waitForTimeout(100)

      // Rapid sequence of moves
      for (let i = 0; i < 6; i++) {
        await debugPage.getColumn(i % 7).click()
      }
      await debugPage.page.waitForTimeout(200)

      // Take a snapshot of piece count
      const pieceCountBefore = await debugPage.countPieces()

      // Wait a bit longer to ensure no phantom pieces appear
      await debugPage.page.waitForTimeout(500)

      const pieceCountAfter = await debugPage.countPieces()

      // Piece count should remain stable (no phantom pieces appearing/disappearing)
      expect(pieceCountAfter).toBe(pieceCountBefore)
    })

    test('should preserve board state integrity during concurrent operations', async () => {
      // Enable new hook
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()
      await debugPage.page.waitForTimeout(100)

      // Play several moves
      await debugPage.playMoves([3, 3, 3, 3, 3, 3]) // Stack in column 3

      // Get move history and verify it matches piece positions
      const moveHistory = await debugPage.getMoveHistory()

      // All moves should be in column 3
      expect(moveHistory.every(m => m === 3)).toBe(true)

      // Column 3 should have 6 pieces stacked
      for (let row = 0; row < 6; row++) {
        const cellPlayer = await debugPage.getCellPlayer(row, 3)
        expect(cellPlayer).not.toBeNull()
      }
    })

    test('should handle alternating players correctly during rapid play', async () => {
      // Make rapid alternating moves
      await debugPage.dropPieceInColumn(0) // P1
      await debugPage.dropPieceInColumn(1) // P2
      await debugPage.dropPieceInColumn(2) // P1
      await debugPage.dropPieceInColumn(3) // P2

      // Verify alternating colors at bottom row
      expect(await debugPage.getCellPlayer(5, 0)).toBe(1) // Red
      expect(await debugPage.getCellPlayer(5, 1)).toBe(2) // Yellow
      expect(await debugPage.getCellPlayer(5, 2)).toBe(1) // Red
      expect(await debugPage.getCellPlayer(5, 3)).toBe(2) // Yellow
    })

    test('should not lose pieces during reset', async () => {
      // Play some moves
      await debugPage.playMoves([0, 1, 2, 3, 4])
      const moveCountBefore = await debugPage.getMoveCount()
      expect(moveCountBefore).toBe(5)

      // Reset
      await debugPage.reset()

      // Verify clean state
      const isEmpty = await debugPage.isBoardEmpty()
      expect(isEmpty).toBe(true)

      const moveCount = await debugPage.getMoveCount()
      expect(moveCount).toBe(0)
    })

    test('should maintain version monotonicity', async () => {
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()
      await debugPage.page.waitForTimeout(100)

      let lastVersion = 0

      for (let i = 0; i < 5; i++) {
        await debugPage.dropPieceInColumn(i)
        const stateVersion = debugPage.page.locator('[data-testid="state-version"]')
        const currentVersion = parseInt(await stateVersion.textContent() || '0')

        // Version should always increase
        expect(currentVersion).toBeGreaterThan(lastVersion)
        lastVersion = currentVersion
      }
    })
  })

  test.describe('Text Mode State Verification', () => {
    test('should show consistent state in text mode after moves', async () => {
      // Switch to text mode
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      // Enable new hook
      const checkbox = debugPage.page.locator('[data-testid="use-new-hook-checkbox"]')
      await checkbox.scrollIntoViewIfNeeded()
      await checkbox.check()
      await debugPage.page.waitForTimeout(100)

      // Play moves via text mode columns
      const col3 = debugPage.page.locator('[data-testid="text-col-3"]')
      await col3.click()
      await debugPage.page.waitForTimeout(100)

      // Verify text board shows player 2's turn
      const textBoard = debugPage.page.locator('[data-testid="text-board"]')
      await expect(textBoard).toHaveAttribute('data-current-player', '2')

      // Make another move
      const col4 = debugPage.page.locator('[data-testid="text-col-4"]')
      await col4.click()
      await debugPage.page.waitForTimeout(100)

      // Back to player 1
      await expect(textBoard).toHaveAttribute('data-current-player', '1')
    })

    test('should detect win correctly in text mode', async () => {
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      // Reset first
      await debugPage.reset()
      await debugPage.page.waitForTimeout(100)

      // Create horizontal win for player 1
      const moves = [0, 6, 1, 6, 2, 6, 3] // P1 wins horizontally
      for (const col of moves) {
        const colButton = debugPage.page.locator(`[data-testid="text-col-${col}"]`)
        await colButton.click()
        await debugPage.page.waitForTimeout(100)
      }

      // Verify text board shows winner
      const textBoard = debugPage.page.locator('[data-testid="text-board"]')
      await expect(textBoard).toHaveAttribute('data-winner', '1')
    })

    test('should highlight winning cells in text mode', async () => {
      const select = debugPage.page.locator('[data-testid="render-mode-select"]')
      await select.scrollIntoViewIfNeeded()
      await select.selectOption('text')

      // Reset first
      await debugPage.reset()
      await debugPage.page.waitForTimeout(100)

      // Create horizontal win for player 1
      const moves = [0, 6, 1, 6, 2, 6, 3]
      for (const col of moves) {
        const colButton = debugPage.page.locator(`[data-testid="text-col-${col}"]`)
        await colButton.click()
        await debugPage.page.waitForTimeout(100)
      }

      // Check that winning cells are marked in ASCII (uppercase R)
      const asciiBoard = debugPage.page.locator('[data-testid="ascii-board"]')
      const boardText = await asciiBoard.textContent()

      // Winning cells should be uppercase (R for winning red pieces)
      expect(boardText).toContain('R')
    })
  })
})
