/**
 * Test chat on production site to reproduce duplicate messages
 */

import { chromium } from 'playwright'

const BASE_URL = process.env.TEST_URL || 'https://makefour.org'

async function testProdChat() {
  console.log(`\n🧪 Testing chat on production: ${BASE_URL}\n`)

  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext()
  const page = await context.newPage()

  // Capture console logs - especially Chat Debug logs
  page.on('console', msg => {
    const text = msg.text()
    if (text.includes('[Chat Debug]') || text.includes('DUPLICATE') || text.includes('BLOCKED') || text.includes('bot') || text.includes('Bot')) {
      console.log(`[BROWSER] ${text}`)
    }
  })

  // Capture network responses for chat API
  page.on('response', async response => {
    if (response.url().includes('/chat')) {
      try {
        const json = await response.json()
        console.log(`[API] ${response.url().split('/').slice(-2).join('/')}: ${JSON.stringify(json).slice(0, 200)}`)
      } catch {}
    }
  })

  try {
    // 1. Login using debug endpoint
    console.log('1. Logging in via debug endpoint...')

    const loginResponse = await fetch(`${BASE_URL}/api/debug/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    })

    if (!loginResponse.ok) {
      throw new Error(`Debug login failed: ${loginResponse.status}`)
    }

    const loginData = await loginResponse.json()
    const sessionToken = loginData.session_token
    console.log(`   Logged in as: ${loginData.user?.username}`)
    console.log(`   Session token: ${sessionToken?.slice(0, 8)}...`)

    if (!sessionToken) {
      throw new Error('No session token in login response')
    }

    // Set in localStorage via page (this is the primary auth mechanism)
    await page.goto(BASE_URL, { waitUntil: 'networkidle' })
    await page.evaluate((token) => {
      localStorage.setItem('makefour_session_token', token)
      console.log('[Test] Set session token in localStorage')
    }, sessionToken)

    // Reload to pick up the auth state
    await page.reload({ waitUntil: 'networkidle' })
    await page.waitForTimeout(1000)

    // Verify we're logged in by checking for Compete link
    const competeLink = await page.locator('a:has-text("Compete")').count()
    console.log(`   Compete link visible: ${competeLink > 0 ? 'YES' : 'NO'}`)

    if (competeLink === 0) {
      await page.screenshot({ path: '/tmp/prod-chat-auth-debug.png', fullPage: true })
      console.log('   Auth debug screenshot: /tmp/prod-chat-auth-debug.png')
      throw new Error('Not logged in after setting session token')
    }

    // 2. Create a debug game
    console.log('\n2. Creating debug game...')

    const gameResponse = await fetch(`${BASE_URL}/api/debug/game`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${sessionToken}`,
      },
    })

    if (!gameResponse.ok) {
      const errText = await gameResponse.text()
      throw new Error(`Debug game creation failed: ${gameResponse.status} - ${errText}`)
    }

    const gameData = await gameResponse.json()
    console.log(`   Game ID: ${gameData.gameId}`)
    console.log(`   Bot: ${gameData.botName}`)

    // 3. Navigate to competitive mode
    console.log('\n3. Navigating to competitive play...')

    // Click on Compete link in nav
    await page.click('text=Compete')
    await page.waitForTimeout(2000)

    // Take screenshot
    await page.screenshot({ path: '/tmp/prod-chat-1-compete.png', fullPage: true })
    console.log('   Screenshot: /tmp/prod-chat-1-compete.png')

    // 4. Start a game via Automatch
    console.log('\n4. Starting game via Automatch...')

    // Look for Automatch button
    const automatchBtn = page.locator('h3:has-text("Automatch")').or(page.locator('text=Automatch'))
    if (await automatchBtn.count() > 0) {
      console.log('   Clicking Automatch...')
      await automatchBtn.first().click()
      await page.waitForTimeout(3000)

      // Screenshot the matchmaking screen
      await page.screenshot({ path: '/tmp/prod-chat-2-matchmaking.png', fullPage: true })
      console.log('   Screenshot: /tmp/prod-chat-2-matchmaking.png')

      // Click "Play Bot Now" if it appears
      const playBotBtn = page.locator('text=Play Bot Now')
      if (await playBotBtn.count() > 0) {
        console.log('   Clicking Play Bot Now...')
        await playBotBtn.click()
        await page.waitForTimeout(3000)
      }
    } else {
      console.log('   Automatch not found, trying other options...')
    }

    // Take screenshot of current state
    await page.screenshot({ path: '/tmp/prod-chat-3-game.png', fullPage: true })
    console.log('   Screenshot: /tmp/prod-chat-3-game.png')

    // 5. Look for chat panel
    console.log('\n5. Looking for chat panel...')

    // Check if there's a chat panel
    const chatPanels = await page.locator('text=Chat').count()
    console.log(`   Found ${chatPanels} chat panel(s)`)

    // 6. Find and interact with chat
    console.log('\n6. Testing chat...')
    const chatInput = page.locator('input[placeholder*="message"]').or(page.locator('input[placeholder*="Type"]'))
    const chatInputCount = await chatInput.count()
    console.log(`   Chat inputs found: ${chatInputCount}`)

    if (chatInputCount > 0) {
      // Count messages before
      const messagesBefore = await page.locator('.rounded-lg.px-3.py-1\\.5').count()
      console.log(`   Messages before send: ${messagesBefore}`)

      // Type and send message
      const testMessage = `test-${Date.now()}`
      console.log(`   Sending: "${testMessage}"`)

      await chatInput.first().fill(testMessage)
      await page.keyboard.press('Enter')

      // Wait for message to appear and debug logs to fire
      await page.waitForTimeout(3000)

      // Count messages after
      const messagesAfter = await page.locator('.rounded-lg.px-3.py-1\\.5').count()
      console.log(`   Messages after send: ${messagesAfter}`)

      // Check for our message
      const ourMessages = await page.locator(`text="${testMessage}"`).count()
      console.log(`   Instances of our message: ${ourMessages}`)

      if (ourMessages > 1) {
        console.log('\n   ❌ DUPLICATE DETECTED!')
      } else if (ourMessages === 1) {
        console.log('\n   ✅ No duplicates - message appears once')
      } else {
        console.log('\n   ⚠️  Message not found')
      }

      // Check for bot response
      console.log('\n   Checking for bot response...')
      await page.waitForTimeout(3000) // Give bot time to respond

      const botMessages = await page.locator('.bg-purple-100, .dark\\:bg-purple-900\\/30').count()
      console.log(`   Bot messages found: ${botMessages}`)

      if (botMessages > 0) {
        console.log('   ✅ Bot responded!')
      } else {
        console.log('   ⚠️  No bot response received')
      }

      // Send a second message to test rapid sends
      console.log('\n7. Testing rapid message send...')
      const testMessage2 = `rapid-${Date.now()}`
      await chatInput.first().fill(testMessage2)
      await page.keyboard.press('Enter')

      // Try to send same message again immediately
      await chatInput.first().fill(testMessage2)
      await page.keyboard.press('Enter')

      await page.waitForTimeout(3000)

      const rapidMessages = await page.locator(`text="${testMessage2}"`).count()
      console.log(`   Instances of rapid message: ${rapidMessages}`)

      if (rapidMessages > 1) {
        console.log('   ❌ DUPLICATE from rapid send!')
      } else if (rapidMessages === 1) {
        console.log('   ✅ Rapid send protection working')
      }

      // Take final screenshot
      await page.screenshot({ path: '/tmp/prod-chat-3-after.png', fullPage: true })
      console.log('   Screenshot: /tmp/prod-chat-3-after.png')
    }

    console.log('\n' + '='.repeat(50))
    console.log('Test complete!')
    console.log('Screenshots saved to /tmp/prod-chat-*.png')

  } catch (error) {
    console.error('\n❌ [ERROR]', error.message)
    try {
      await page.screenshot({ path: '/tmp/prod-chat-error.png', fullPage: true })
      console.log('Error screenshot: /tmp/prod-chat-error.png')
    } catch {}
  } finally {
    await browser.close()
  }
}

testProdChat()
