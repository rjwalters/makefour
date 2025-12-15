import { useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'

interface ExportFilters {
  dateFrom?: string
  dateTo?: string
  minMoves?: number
  maxMoves?: number
  outcomes?: ('win' | 'loss' | 'draw')[]
  opponentTypes?: ('human' | 'ai')[]
  aiDifficulties?: ('beginner' | 'intermediate' | 'expert' | 'perfect')[]
  limit?: number
}

interface ExportModalProps {
  isOpen: boolean
  onClose: () => void
  onExport: (format: 'json' | 'pgn', filters: ExportFilters) => Promise<void>
  totalGames: number
}

export default function ExportModal({ isOpen, onClose, onExport, totalGames }: ExportModalProps) {
  const [format, setFormat] = useState<'json' | 'pgn'>('json')
  const [isExporting, setIsExporting] = useState(false)
  const [exportError, setExportError] = useState<string | null>(null)
  const [filters, setFilters] = useState<ExportFilters>({
    outcomes: [],
    opponentTypes: [],
    aiDifficulties: [],
    limit: 1000,
  })

  const handleExport = async () => {
    setIsExporting(true)
    setExportError(null)
    try {
      await onExport(format, filters)
      onClose()
    } catch (err) {
      setExportError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setIsExporting(false)
    }
  }

  const toggleOutcome = (outcome: 'win' | 'loss' | 'draw') => {
    setFilters((prev) => {
      const outcomes = prev.outcomes || []
      if (outcomes.includes(outcome)) {
        return { ...prev, outcomes: outcomes.filter((o) => o !== outcome) }
      }
      return { ...prev, outcomes: [...outcomes, outcome] }
    })
  }

  const toggleOpponentType = (type: 'human' | 'ai') => {
    setFilters((prev) => {
      const types = prev.opponentTypes || []
      if (types.includes(type)) {
        return { ...prev, opponentTypes: types.filter((t) => t !== type) }
      }
      return { ...prev, opponentTypes: [...types, type] }
    })
  }

  const toggleAiDifficulty = (difficulty: 'beginner' | 'intermediate' | 'expert' | 'perfect') => {
    setFilters((prev) => {
      const difficulties = prev.aiDifficulties || []
      if (difficulties.includes(difficulty)) {
        return { ...prev, aiDifficulties: difficulties.filter((d) => d !== difficulty) }
      }
      return { ...prev, aiDifficulties: [...difficulties, difficulty] }
    })
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-lg max-h-[90vh] overflow-y-auto">
        <CardHeader>
          <CardTitle>Export Games</CardTitle>
          <CardDescription>
            Export your game data for training or analysis ({totalGames} games available)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Format Selection */}
          <fieldset className="border-0 p-0 m-0">
            <legend className="block text-sm font-medium mb-2">Export Format</legend>
            <div className="flex gap-2">
              <Button
                variant={format === 'json' ? 'default' : 'outline'}
                onClick={() => setFormat('json')}
                size="sm"
              >
                JSON
              </Button>
              <Button
                variant={format === 'pgn' ? 'default' : 'outline'}
                onClick={() => setFormat('pgn')}
                size="sm"
              >
                PGN
              </Button>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {format === 'json'
                ? 'Machine-readable format for training neural networks'
                : 'Human-readable format similar to chess PGN notation'}
            </p>
          </fieldset>

          {/* Date Range */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="export-from-date" className="block text-sm font-medium mb-2">From Date</label>
              <Input
                id="export-from-date"
                type="date"
                value={filters.dateFrom || ''}
                onChange={(e) => setFilters({ ...filters, dateFrom: e.target.value || undefined })}
              />
            </div>
            <div>
              <label htmlFor="export-to-date" className="block text-sm font-medium mb-2">To Date</label>
              <Input
                id="export-to-date"
                type="date"
                value={filters.dateTo || ''}
                onChange={(e) => setFilters({ ...filters, dateTo: e.target.value || undefined })}
              />
            </div>
          </div>

          {/* Move Count Range */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="export-min-moves" className="block text-sm font-medium mb-2">Min Moves</label>
              <Input
                id="export-min-moves"
                type="number"
                min={1}
                placeholder="1"
                value={filters.minMoves || ''}
                onChange={(e) =>
                  setFilters({ ...filters, minMoves: e.target.value ? parseInt(e.target.value, 10) : undefined })
                }
              />
            </div>
            <div>
              <label htmlFor="export-max-moves" className="block text-sm font-medium mb-2">Max Moves</label>
              <Input
                id="export-max-moves"
                type="number"
                min={1}
                placeholder="42"
                value={filters.maxMoves || ''}
                onChange={(e) =>
                  setFilters({ ...filters, maxMoves: e.target.value ? parseInt(e.target.value, 10) : undefined })
                }
              />
            </div>
          </div>

          {/* Outcome Filter */}
          <fieldset className="border-0 p-0 m-0">
            <legend className="block text-sm font-medium mb-2">Outcomes</legend>
            <div className="flex gap-2 flex-wrap">
              {(['win', 'loss', 'draw'] as const).map((outcome) => (
                <Button
                  key={outcome}
                  variant={filters.outcomes?.includes(outcome) ? 'default' : 'outline'}
                  onClick={() => toggleOutcome(outcome)}
                  size="sm"
                >
                  {outcome.charAt(0).toUpperCase() + outcome.slice(1)}
                </Button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Leave empty to include all outcomes
            </p>
          </fieldset>

          {/* Opponent Type Filter */}
          <fieldset className="border-0 p-0 m-0">
            <legend className="block text-sm font-medium mb-2">Opponent Type</legend>
            <div className="flex gap-2 flex-wrap">
              <Button
                variant={filters.opponentTypes?.includes('ai') ? 'default' : 'outline'}
                onClick={() => toggleOpponentType('ai')}
                size="sm"
              >
                AI Games
              </Button>
              <Button
                variant={filters.opponentTypes?.includes('human') ? 'default' : 'outline'}
                onClick={() => toggleOpponentType('human')}
                size="sm"
              >
                Hotseat Games
              </Button>
            </div>
          </fieldset>

          {/* AI Difficulty Filter */}
          <fieldset className="border-0 p-0 m-0">
            <legend className="block text-sm font-medium mb-2">AI Difficulty</legend>
            <div className="flex gap-2 flex-wrap">
              {(['beginner', 'intermediate', 'expert', 'perfect'] as const).map((difficulty) => (
                <Button
                  key={difficulty}
                  variant={filters.aiDifficulties?.includes(difficulty) ? 'default' : 'outline'}
                  onClick={() => toggleAiDifficulty(difficulty)}
                  size="sm"
                >
                  {difficulty.charAt(0).toUpperCase() + difficulty.slice(1)}
                </Button>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Only applies to AI games
            </p>
          </fieldset>

          {/* Limit */}
          <div>
            <label htmlFor="export-limit" className="block text-sm font-medium mb-2">Max Games to Export</label>
            <Input
              id="export-limit"
              type="number"
              min={1}
              max={10000}
              value={filters.limit || 1000}
              onChange={(e) =>
                setFilters({ ...filters, limit: Math.min(10000, parseInt(e.target.value, 10) || 1000) })
              }
            />
            <p className="text-xs text-muted-foreground mt-1">Maximum 10,000 games per export</p>
          </div>

          {/* Error Display */}
          {exportError && (
            <div className="p-3 rounded-md bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm">
              {exportError}
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 justify-end pt-4 border-t">
            <Button variant="outline" onClick={onClose} disabled={isExporting}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={isExporting}>
              {isExporting ? 'Exporting...' : `Export as ${format.toUpperCase()}`}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
