/**
 * Custom React hook for game sound effects using Web Audio API
 *
 * Generates sounds programmatically to minimize bundle size.
 * Sounds are designed to be pleasant and unobtrusive.
 */

import { useCallback, useRef, useEffect, useState } from 'react'

export interface SoundSettings {
  enabled: boolean
  volume: number // 0-100
}

const DEFAULT_SOUND_SETTINGS: SoundSettings = {
  enabled: true,
  volume: 50,
}

const STORAGE_KEY = 'makefour-sound-settings'

function loadSoundSettings(): SoundSettings {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      const parsed = JSON.parse(saved)
      return {
        enabled: parsed.enabled ?? DEFAULT_SOUND_SETTINGS.enabled,
        volume: typeof parsed.volume === 'number' ? parsed.volume : DEFAULT_SOUND_SETTINGS.volume,
      }
    }
  } catch {
    // Ignore parse errors
  }
  return DEFAULT_SOUND_SETTINGS
}

function saveSoundSettings(settings: SoundSettings) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings))
}

export function useSounds() {
  const audioContextRef = useRef<AudioContext | null>(null)
  const [settings, setSettings] = useState<SoundSettings>(loadSoundSettings)

  // Initialize AudioContext on first user interaction (required by browsers)
  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext()
    }
    // Resume if suspended (browser autoplay policy)
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume()
    }
    return audioContextRef.current
  }, [])

  // Save settings when they change
  useEffect(() => {
    saveSoundSettings(settings)
  }, [settings])

  // Get normalized volume (0-1)
  const getVolume = useCallback(() => {
    return settings.volume / 100
  }, [settings.volume])

  /**
   * Play a simple tone
   */
  const playTone = useCallback(
    (frequency: number, duration: number, type: OscillatorType = 'sine', gainMultiplier = 1) => {
      if (!settings.enabled) return

      const ctx = getAudioContext()
      const oscillator = ctx.createOscillator()
      const gainNode = ctx.createGain()

      oscillator.connect(gainNode)
      gainNode.connect(ctx.destination)

      oscillator.type = type
      oscillator.frequency.setValueAtTime(frequency, ctx.currentTime)

      const volume = getVolume() * gainMultiplier * 0.3 // Keep overall volume reasonable
      gainNode.gain.setValueAtTime(volume, ctx.currentTime)
      gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration)

      oscillator.start(ctx.currentTime)
      oscillator.stop(ctx.currentTime + duration)
    },
    [settings.enabled, getAudioContext, getVolume]
  )

  /**
   * Play a sequence of tones
   */
  const playSequence = useCallback(
    (
      notes: Array<{ frequency: number; duration: number; delay: number }>,
      type: OscillatorType = 'sine',
      gainMultiplier = 1
    ) => {
      if (!settings.enabled) return

      const ctx = getAudioContext()
      const volume = getVolume() * gainMultiplier * 0.3

      notes.forEach(({ frequency, duration, delay }) => {
        const oscillator = ctx.createOscillator()
        const gainNode = ctx.createGain()

        oscillator.connect(gainNode)
        gainNode.connect(ctx.destination)

        oscillator.type = type
        oscillator.frequency.setValueAtTime(frequency, ctx.currentTime + delay)

        gainNode.gain.setValueAtTime(0, ctx.currentTime + delay)
        gainNode.gain.linearRampToValueAtTime(volume, ctx.currentTime + delay + 0.01)
        gainNode.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + delay + duration)

        oscillator.start(ctx.currentTime + delay)
        oscillator.stop(ctx.currentTime + delay + duration)
      })
    },
    [settings.enabled, getAudioContext, getVolume]
  )

  /**
   * Piece drop sound - satisfying "click" when piece lands
   */
  const playPieceDrop = useCallback(() => {
    // Short descending tone like a piece clicking into place
    playSequence(
      [
        { frequency: 600, duration: 0.08, delay: 0 },
        { frequency: 400, duration: 0.12, delay: 0.03 },
      ],
      'sine',
      0.8
    )
  }, [playSequence])

  /**
   * Your turn notification sound
   */
  const playYourTurn = useCallback(() => {
    // Gentle rising notification
    playSequence(
      [
        { frequency: 440, duration: 0.15, delay: 0 },
        { frequency: 554, duration: 0.2, delay: 0.1 },
      ],
      'sine',
      0.6
    )
  }, [playSequence])

  /**
   * Victory fanfare sound
   */
  const playWin = useCallback(() => {
    // Ascending major arpeggio (C-E-G-C)
    playSequence(
      [
        { frequency: 523, duration: 0.2, delay: 0 }, // C5
        { frequency: 659, duration: 0.2, delay: 0.15 }, // E5
        { frequency: 784, duration: 0.2, delay: 0.3 }, // G5
        { frequency: 1047, duration: 0.4, delay: 0.45 }, // C6
      ],
      'sine',
      1
    )
  }, [playSequence])

  /**
   * Defeat sound
   */
  const playLose = useCallback(() => {
    // Descending minor tones
    playSequence(
      [
        { frequency: 400, duration: 0.25, delay: 0 },
        { frequency: 350, duration: 0.25, delay: 0.2 },
        { frequency: 300, duration: 0.4, delay: 0.4 },
      ],
      'sine',
      0.7
    )
  }, [playSequence])

  /**
   * Draw sound - neutral end-game tone
   */
  const playDraw = useCallback(() => {
    // Two neutral tones
    playSequence(
      [
        { frequency: 440, duration: 0.2, delay: 0 },
        { frequency: 440, duration: 0.3, delay: 0.25 },
      ],
      'sine',
      0.6
    )
  }, [playSequence])

  /**
   * Invalid move / error sound
   */
  const playInvalidMove = useCallback(() => {
    // Short harsh buzz
    playTone(200, 0.15, 'square', 0.4)
  }, [playTone])

  /**
   * Game start sound
   */
  const playGameStart = useCallback(() => {
    // Two-tone chime (like a ready sound)
    playSequence(
      [
        { frequency: 523, duration: 0.15, delay: 0 }, // C5
        { frequency: 784, duration: 0.25, delay: 0.12 }, // G5
      ],
      'sine',
      0.7
    )
  }, [playSequence])

  /**
   * Match found sound (for online matchmaking)
   */
  const playMatchFound = useCallback(() => {
    // Excited ascending tones
    playSequence(
      [
        { frequency: 440, duration: 0.12, delay: 0 },
        { frequency: 554, duration: 0.12, delay: 0.1 },
        { frequency: 659, duration: 0.12, delay: 0.2 },
        { frequency: 880, duration: 0.2, delay: 0.3 },
      ],
      'sine',
      0.8
    )
  }, [playSequence])

  /**
   * Update sound settings
   */
  const updateSettings = useCallback((newSettings: Partial<SoundSettings>) => {
    setSettings((prev) => ({ ...prev, ...newSettings }))
  }, [])

  /**
   * Toggle sound on/off
   */
  const toggleSound = useCallback(() => {
    setSettings((prev) => ({ ...prev, enabled: !prev.enabled }))
  }, [])

  return {
    // Settings
    settings,
    updateSettings,
    toggleSound,

    // Sound effects
    playPieceDrop,
    playYourTurn,
    playWin,
    playLose,
    playDraw,
    playInvalidMove,
    playGameStart,
    playMatchFound,
  }
}
