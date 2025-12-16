/**
 * AI Engines Index
 *
 * Registers all available AI engines with the global registry.
 * Import this module to ensure engines are registered before use.
 */

import { engineRegistry } from '../ai-engine'
import { minimaxEngine } from './minimax-engine'
import { aggressiveMinimaxEngine } from './aggressive-minimax-engine'
import { deepMinimaxEngine } from './deep-minimax-engine'
import { neuralEngine } from './neural-engine'
import { claimEvenEngine } from './claimeven-engine'
import { parityEngine } from './parity-engine'
import { threatPairsEngine } from './threat-pairs-engine'

// Register all available engines
engineRegistry.register(minimaxEngine, true)
engineRegistry.register(aggressiveMinimaxEngine, true)
engineRegistry.register(deepMinimaxEngine, true)
engineRegistry.register(neuralEngine, true) // Simulated mode always available
engineRegistry.register(claimEvenEngine, true) // 2swap claimeven strategy
engineRegistry.register(parityEngine, true) // 2swap parity strategy
engineRegistry.register(threatPairsEngine, true) // 2swap threat pairs strategy

// Set standard minimax as the default engine
engineRegistry.setDefault('minimax')

// Re-export engines for direct access if needed
export {
  minimaxEngine,
  aggressiveMinimaxEngine,
  deepMinimaxEngine,
  neuralEngine,
  claimEvenEngine,
  parityEngine,
  threatPairsEngine,
}

// Re-export registry utilities
export { engineRegistry }
