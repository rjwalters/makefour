/**
 * Sound toggle button with optional volume slider
 *
 * Provides a quick way to toggle sounds on/off with an expandable volume control.
 */

import { useState, useRef, useEffect } from 'react'
import { Button } from './ui/button'
import type { SoundSettings } from '../hooks/useSounds'

interface SoundToggleProps {
  settings: SoundSettings
  onToggle: () => void
  onVolumeChange: (volume: number) => void
}

export default function SoundToggle({ settings, onToggle, onVolumeChange }: SoundToggleProps) {
  const [showVolume, setShowVolume] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  // Close volume slider when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setShowVolume(false)
      }
    }

    if (showVolume) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [showVolume])

  const getIcon = () => {
    if (!settings.enabled) return '🔇'
    if (settings.volume === 0) return '🔇'
    if (settings.volume < 33) return '🔈'
    if (settings.volume < 66) return '🔉'
    return '🔊'
  }

  const handleButtonClick = () => {
    onToggle()
  }

  const handleButtonRightClick = (e: React.MouseEvent) => {
    e.preventDefault()
    setShowVolume(!showVolume)
  }

  return (
    <div ref={containerRef} className="relative">
      <Button
        variant="outline"
        size="sm"
        onClick={handleButtonClick}
        onContextMenu={handleButtonRightClick}
        className="gap-2"
        title={settings.enabled ? 'Sound on (right-click for volume)' : 'Sound off (right-click for volume)'}
      >
        <span>{getIcon()}</span>
        <span className="hidden sm:inline">{settings.enabled ? 'Sound' : 'Muted'}</span>
      </Button>

      {/* Volume slider dropdown */}
      {showVolume && (
        <div className="absolute right-0 top-full mt-2 z-50 bg-background border rounded-lg shadow-lg p-3 min-w-[150px]">
          <div className="flex items-center gap-2">
            <span className="text-sm">🔈</span>
            <input
              type="range"
              min="0"
              max="100"
              value={settings.volume}
              onChange={(e) => onVolumeChange(Number(e.target.value))}
              className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            />
            <span className="text-sm">🔊</span>
          </div>
          <div className="text-center text-xs text-muted-foreground mt-1">
            {settings.volume}%
          </div>
        </div>
      )}
    </div>
  )
}
