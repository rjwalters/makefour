/**
 * Playwright test for the debug game render page
 *
 * Tests:
 * - Page loads correctly
 * - Both players can be played by clicking
 * - Debug logging works
 * - Game state updates correctly
 * - Win detection works
 * - Undo functionality works
 */

import { chromium } from 'playwright'

const BASE_URL = process.env.TEST_URL || 'https://1e4c3dd4.makefour.pages.dev'

async function testGameRender() {
  console.log(`\n🧪 Testing Debug Game Render Page\n`)
  console.log(`URL: ${BASE_URL}/debug/game-render\n`)

  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext()
  const page = await context.newPage()

  try {
    // 1. Navigate to debug page
    console.log('1. Loading debug page...')
    await page.goto(`${BASE_URL}/debug/game-render`, { waitUntil: 'networkidle', timeout: 30000 })

    const pageLoaded = await page.locator('[data-testid="debug-game-render-page"]').count() > 0
    console.log(`   Page loaded: ${pageLoaded ? 'YES' : 'NO'}`)

    if (!pageLoaded) {
      throw new Error('Debug page did not load')
    }

    // 2. Check initial state
    console.log('\n2. Checking initial state...')
    const currentPlayer = await page.locator('[data-testid="current-player"]').textContent()
    const moveCount = await page.locator('[data-testid="move-count"]').textContent()
    const winner = await page.locator('[data-testid="winner"]').textContent()

    console.log(`   Current player: ${currentPlayer}`)
    console.log(`   Move count: ${moveCount}`)
    console.log(`   Winner: ${winner}`)

    // 3. Play a few moves
    console.log('\n3. Playing moves...')

    // Player 1 (Red) - column 3
    await page.locator('[data-testid="column-3"]').click()
    await page.waitForTimeout(100)
    let newMoveCount = await page.locator('[data-testid="move-count"]').textContent()
    console.log(`   After move 1: ${newMoveCount} moves`)

    // Player 2 (Yellow) - column 4
    await page.locator('[data-testid="column-4"]').click()
    await page.waitForTimeout(100)
    newMoveCount = await page.locator('[data-testid="move-count"]').textContent()
    console.log(`   After move 2: ${newMoveCount} moves`)

    // Player 1 (Red) - column 3 again
    await page.locator('[data-testid="column-3"]').click()
    await page.waitForTimeout(100)
    newMoveCount = await page.locator('[data-testid="move-count"]').textContent()
    console.log(`   After move 3: ${newMoveCount} moves`)

    // 4. Check debug log
    console.log('\n4. Checking debug log...')
    const debugLog = await page.locator('[data-testid="debug-log"]').inputValue()
    const logLines = debugLog.split('\n').filter(l => l.trim())
    console.log(`   Log lines: ${logLines.length}`)

    const hasRenderLogs = debugLog.includes('[RENDER   ]')
    const hasStateLogs = debugLog.includes('[STATE    ]')
    const hasClickLogs = debugLog.includes('[CLICK    ]')
    const hasMoveLogs = debugLog.includes('[MOVE     ]')

    console.log(`   Has RENDER logs: ${hasRenderLogs}`)
    console.log(`   Has STATE logs: ${hasStateLogs}`)
    console.log(`   Has CLICK logs: ${hasClickLogs}`)
    console.log(`   Has MOVE logs: ${hasMoveLogs}`)

    // 5. Test undo
    console.log('\n5. Testing undo...')
    const moveCountBefore = await page.locator('[data-testid="move-count"]').textContent()
    await page.locator('[data-testid="undo-btn"]').click()
    await page.waitForTimeout(100)
    const moveCountAfter = await page.locator('[data-testid="move-count"]').textContent()
    console.log(`   Move count before undo: ${moveCountBefore}`)
    console.log(`   Move count after undo: ${moveCountAfter}`)
    console.log(`   Undo works: ${parseInt(moveCountAfter) === parseInt(moveCountBefore) - 1}`)

    // 6. Test reset
    console.log('\n6. Testing reset...')
    await page.locator('[data-testid="reset-btn"]').click()
    await page.waitForTimeout(100)
    const resetMoveCount = await page.locator('[data-testid="move-count"]').textContent()
    console.log(`   Move count after reset: ${resetMoveCount}`)
    console.log(`   Reset works: ${resetMoveCount === '0'}`)

    // 7. Play to win (vertical win)
    console.log('\n7. Playing to win (vertical)...')
    // Red plays column 0: moves 1, 3, 5, 7
    // Yellow plays column 1: moves 2, 4, 6

    for (let i = 0; i < 4; i++) {
      await page.locator('[data-testid="column-0"]').click() // Red
      await page.waitForTimeout(50)
      if (i < 3) {
        await page.locator('[data-testid="column-1"]').click() // Yellow
        await page.waitForTimeout(50)
      }
    }

    const finalWinner = await page.locator('[data-testid="winner"]').textContent()
    console.log(`   Final winner: ${finalWinner}`)

    // Check for win detection in logs
    const finalLog = await page.locator('[data-testid="debug-log"]').inputValue()
    const hasWinLog = finalLog.includes('[WIN      ]')
    console.log(`   Has WIN log: ${hasWinLog}`)

    // 8. Check winning cells are highlighted
    const winningCells = await page.locator('[data-winning="true"]').count()
    console.log(`   Winning cells highlighted: ${winningCells}`)

    // Screenshot
    await page.screenshot({ path: '/tmp/debug-game-render.png', fullPage: true })
    console.log('\n   Screenshot: /tmp/debug-game-render.png')

    // 9. Summary
    console.log('\n' + '='.repeat(50))
    console.log('\n📊 Test Results:')
    console.log(`   - Page loads: ${pageLoaded ? '✅' : '❌'}`)
    console.log(`   - Moves work: ${parseInt(newMoveCount) > 0 ? '✅' : '❌'}`)
    console.log(`   - Debug logging: ${hasRenderLogs && hasStateLogs && hasClickLogs ? '✅' : '❌'}`)
    console.log(`   - Undo works: ${parseInt(moveCountAfter) === parseInt(moveCountBefore) - 1 ? '✅' : '❌'}`)
    console.log(`   - Reset works: ${resetMoveCount === '0' ? '✅' : '❌'}`)
    console.log(`   - Win detection: ${finalWinner?.includes('1') ? '✅' : '❌'}`)
    console.log(`   - Win logging: ${hasWinLog ? '✅' : '❌'}`)
    console.log(`   - Win highlighting: ${winningCells === 4 ? '✅' : '❌'}`)

  } catch (error) {
    console.error('\n❌ Test Error:', error.message)
    await page.screenshot({ path: '/tmp/debug-game-render-error.png', fullPage: true })
    console.log('Error screenshot: /tmp/debug-game-render-error.png')
    process.exit(1)
  } finally {
    await browser.close()
  }
}

testGameRender()
