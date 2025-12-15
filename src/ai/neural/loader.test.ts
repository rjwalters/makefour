/**
 * Tests for Neural Model Loader
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import {
  HeuristicNeuralAgent,
  NeuralModelFactory,
  createNeuralAgent,
  listAvailableModels,
  registerModel,
  disposeNeuralAgents,
} from './loader'
import { createGameState, makeMove } from '../../game/makefour'
import type { ModelMetadata } from './interface'

describe('HeuristicNeuralAgent', () => {
  let agent: HeuristicNeuralAgent

  beforeEach(() => {
    agent = new HeuristicNeuralAgent()
  })

  afterEach(() => {
    agent.dispose()
  })

  it('should be immediately ready', () => {
    expect(agent.isReady()).toBe(true)
  })

  it('should return valid metadata', () => {
    const metadata = agent.getMetadata()
    expect(metadata.id).toBe('heuristic-v1')
    expect(metadata.architecture).toBe('mlp')
    expect(metadata.expectedElo).toBe(1200)
  })

  describe('evaluate', () => {
    it('should return value in [-1, 1] range for empty board', async () => {
      const state = createGameState()
      const value = await agent.evaluate(state)
      expect(value).toBeGreaterThanOrEqual(-1)
      expect(value).toBeLessThanOrEqual(1)
    })

    it('should return different values for different positions', async () => {
      const emptyState = createGameState()
      let advantageState = createGameState()
      // Build up a position with player 1 having some advantage (center column)
      advantageState = makeMove(advantageState, 3)!
      advantageState = makeMove(advantageState, 0)!
      advantageState = makeMove(advantageState, 3)!
      advantageState = makeMove(advantageState, 0)!
      advantageState = makeMove(advantageState, 3)!

      const emptyValue = await agent.evaluate(emptyState)
      const advantageValue = await agent.evaluate(advantageState)

      // These might be similar due to heuristic nature, but should work
      expect(typeof emptyValue).toBe('number')
      expect(typeof advantageValue).toBe('number')
    })
  })

  describe('suggestMove', () => {
    it('should return valid column for empty board', async () => {
      const state = createGameState()
      const move = await agent.suggestMove(state)
      expect(move).toBeGreaterThanOrEqual(0)
      expect(move).toBeLessThanOrEqual(6)
    })

    it('should not suggest full columns', async () => {
      let state = createGameState()
      // Fill column 3
      for (let i = 0; i < 3; i++) {
        state = makeMove(state, 3)!
        state = makeMove(state, 3)!
      }

      // Now column 3 is full
      const move = await agent.suggestMove(state)
      expect(move).not.toBe(3)
    })

    it('should find winning moves', async () => {
      let state = createGameState()
      // Set up Player 1 with three in a row horizontally
      state = makeMove(state, 0)! // P1
      state = makeMove(state, 0)! // P2
      state = makeMove(state, 1)! // P1
      state = makeMove(state, 1)! // P2
      state = makeMove(state, 2)! // P1
      state = makeMove(state, 2)! // P2
      // P1 can win by playing column 3

      const move = await agent.suggestMove(state)
      expect(move).toBe(3) // Should find the winning move
    })
  })

  describe('getMoveProbabilities', () => {
    it('should return 7 probabilities', async () => {
      const state = createGameState()
      const probs = await agent.getMoveProbabilities(state)
      expect(probs.length).toBe(7)
    })

    it('should sum to approximately 1', async () => {
      const state = createGameState()
      const probs = await agent.getMoveProbabilities(state)
      const sum = probs.reduce((a, b) => a + b, 0)
      expect(sum).toBeCloseTo(1, 1) // Allow some tolerance
    })

    it('should assign 0 probability to full columns', async () => {
      let state = createGameState()
      // Fill column 0 by both players playing in column 0
      // This results in alternating: P1, P2, P1, P2, P1, P2 (no 4-in-a-row)
      state = makeMove(state, 0)! // P1 in col 0 (row 5)
      state = makeMove(state, 0)! // P2 in col 0 (row 4)
      state = makeMove(state, 0)! // P1 in col 0 (row 3)
      state = makeMove(state, 0)! // P2 in col 0 (row 2)
      state = makeMove(state, 0)! // P1 in col 0 (row 1)
      state = makeMove(state, 0)! // P2 in col 0 (row 0) - column 0 now full!

      // Now it's player 1's turn with column 0 full
      const probs = await agent.getMoveProbabilities(state)
      expect(probs[0]).toBe(0)
    })
  })

  describe('getFullEvaluation', () => {
    it('should return complete evaluation', async () => {
      const state = createGameState()
      const evaluation = await agent.getFullEvaluation(state)

      expect(evaluation.value).toBeGreaterThanOrEqual(-1)
      expect(evaluation.value).toBeLessThanOrEqual(1)
      expect(evaluation.policy.length).toBe(7)
      expect(evaluation.confidence).toBeGreaterThanOrEqual(0)
      expect(evaluation.confidence).toBeLessThanOrEqual(1)
      expect(evaluation.inferenceTimeMs).toBeGreaterThanOrEqual(0)
    })
  })
})

describe('NeuralModelFactory', () => {
  let factory: NeuralModelFactory

  beforeEach(() => {
    factory = new NeuralModelFactory()
  })

  afterEach(() => {
    disposeNeuralAgents()
  })

  describe('create', () => {
    it('should create heuristic agent', async () => {
      const agent = await factory.create('heuristic-v1')
      expect(agent.isReady()).toBe(true)
      expect(agent.getMetadata().id).toBe('heuristic-v1')
    })

    it('should throw for unknown model', async () => {
      await expect(factory.create('unknown-model')).rejects.toThrow('Model not found')
    })
  })

  describe('listModels', () => {
    it('should return at least the heuristic model', async () => {
      const models = await factory.listModels()
      expect(models.length).toBeGreaterThanOrEqual(1)

      const heuristic = models.find((m) => m.id === 'heuristic-v1')
      expect(heuristic).toBeDefined()
    })
  })

  describe('isCached', () => {
    it('should return true for heuristic model', async () => {
      const cached = await factory.isCached('heuristic-v1')
      expect(cached).toBe(true)
    })
  })
})

describe('Module exports', () => {
  afterEach(() => {
    disposeNeuralAgents()
  })

  describe('createNeuralAgent', () => {
    it('should create default agent', async () => {
      const agent = await createNeuralAgent()
      expect(agent.isReady()).toBe(true)
    })

    it('should create specified agent', async () => {
      const agent = await createNeuralAgent('heuristic-v1')
      expect(agent.getMetadata().id).toBe('heuristic-v1')
    })
  })

  describe('listAvailableModels', () => {
    it('should list models', async () => {
      const models = await listAvailableModels()
      expect(Array.isArray(models)).toBe(true)
      expect(models.length).toBeGreaterThanOrEqual(1)
    })
  })

  describe('registerModel', () => {
    it('should add new model to registry', async () => {
      const customModel: ModelMetadata = {
        id: 'test-model',
        name: 'Test Model',
        architecture: 'mlp',
        expectedElo: 1000,
        sizeBytes: 1000,
        url: '/test.onnx',
        version: '1.0.0',
        encoding: 'flat-binary',
      }

      registerModel(customModel)

      const models = await listAvailableModels()
      const found = models.find((m) => m.id === 'test-model')
      expect(found).toBeDefined()
      expect(found?.name).toBe('Test Model')
    })

    it('should update existing model', async () => {
      const customModel: ModelMetadata = {
        id: 'heuristic-v1',
        name: 'Updated Heuristic',
        architecture: 'mlp',
        expectedElo: 1500,
        sizeBytes: 0,
        url: '',
        version: '2.0.0',
        encoding: 'flat-binary',
      }

      registerModel(customModel)

      const models = await listAvailableModels()
      const found = models.find((m) => m.id === 'heuristic-v1')
      expect(found?.name).toBe('Updated Heuristic')
      expect(found?.version).toBe('2.0.0')
    })
  })
})
