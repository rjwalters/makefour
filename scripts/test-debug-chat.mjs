import { chromium } from 'playwright'

const BASE_URL = process.env.TEST_URL || 'https://5bd2f0a8.makefour.pages.dev'

async function testDebugChat() {
  console.log(`Testing debug chat page at ${BASE_URL}/debug/chat\n`)

  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext()
  const page = await context.newPage()

  try {
    // 1. Navigate to debug chat page
    console.log('1. Navigating to /debug/chat...')
    await page.goto(`${BASE_URL}/debug/chat`, { waitUntil: 'networkidle', timeout: 30000 })

    if (!page.url().includes('/debug/chat')) {
      throw new Error('Page redirected away from /debug/chat!')
    }

    await page.waitForSelector('[data-testid="debug-chat-page"]', { timeout: 10000 })
    console.log('   Page rendered successfully')

    // 2. Login
    console.log('\n2. Logging in as debug user...')
    const loginBtn = page.locator('[data-testid="debug-login-btn"]')
    if (await loginBtn.count() > 0) {
      await loginBtn.click()
      await page.waitForLoadState('networkidle')
      await page.waitForSelector('[data-testid="debug-chat-page"]', { timeout: 10000 })
    }

    // Wait for auth state to update
    await page.waitForFunction(() => {
      const el = document.querySelector('[data-testid="auth-status"]')
      return el && el.textContent.includes('YES')
    }, { timeout: 10000 })

    const authStatus = await page.locator('[data-testid="auth-status"]').textContent()
    console.log(`   ${authStatus}`)

    // 3. Create debug game
    console.log('\n3. Creating debug game...')
    await page.locator('[data-testid="create-game-btn"]').click()

    // Wait for game ID to populate
    await page.waitForFunction(() => {
      const input = document.querySelector('[data-testid="game-id-input"]')
      return input && input.value && input.value.length > 10
    }, { timeout: 10000 })

    const gameId = await page.locator('[data-testid="game-id-input"]').inputValue()
    console.log(`   Game ID: ${gameId}`)

    // 4. Connect to game
    console.log('\n4. Connecting to game chat...')
    await page.locator('[data-testid="connect-btn"]').click()
    await page.waitForTimeout(1000)

    const connectionStatus = await page.locator('[data-testid="connection-status"]').textContent()
    console.log(`   ${connectionStatus}`)

    // 5. Send a test message
    console.log('\n5. Sending test message...')
    await page.locator('[data-testid="message-input"]').fill('Hello from Playwright test!')
    await page.locator('[data-testid="send-btn"]').click()
    await page.waitForTimeout(2000)

    // 6. Check messages
    const messagesJson = await page.locator('#messages-json').inputValue()
    const messages = JSON.parse(messagesJson)
    console.log(`\n6. Messages received: ${messages.length}`)
    for (const msg of messages) {
      console.log(`   [${msg.sender_type}] ${msg.content}`)
    }

    // 7. Check for duplicates
    const ids = messages.map(m => m.id)
    const uniqueIds = new Set(ids)
    if (ids.length !== uniqueIds.size) {
      console.log('\n   [WARN] DUPLICATE MESSAGES DETECTED!')
    } else {
      console.log('\n7. No duplicate messages detected')
    }

    // 8. Trigger bot reaction
    console.log('\n8. Triggering bot reaction...')
    await page.locator('[data-testid="trigger-bot-btn"]').click()
    await page.waitForTimeout(3000)

    // 9. Check final messages
    const finalMessagesJson = await page.locator('#messages-json').inputValue()
    const finalMessages = JSON.parse(finalMessagesJson)
    console.log(`\n9. Final messages: ${finalMessages.length}`)
    for (const msg of finalMessages) {
      console.log(`   [${msg.sender_type}] ${msg.content}`)
    }

    // 10. Get debug log
    const debugLog = await page.locator('#debug-log').inputValue()
    console.log('\n10. Debug log:')
    console.log('---')
    console.log(debugLog)
    console.log('---')

    // Screenshot
    await page.screenshot({ path: '/tmp/debug-chat-full-test.png', fullPage: true })
    console.log('\n11. Screenshot saved to /tmp/debug-chat-full-test.png')

    console.log('\n[SUCCESS] Full debug chat test completed!')

  } catch (error) {
    console.error('\n[ERROR]', error.message)

    try {
      const debugLog = await page.locator('#debug-log').inputValue()
      console.log('\nDebug log at error:')
      console.log(debugLog)
    } catch {}

    await page.screenshot({ path: '/tmp/debug-chat-error.png', fullPage: true })
    process.exit(1)
  } finally {
    await browser.close()
  }
}

testDebugChat()
