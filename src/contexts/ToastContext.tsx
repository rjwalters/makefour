import type React from 'react'
import { createContext, useContext, useState, useCallback } from 'react'

export type ToastType = 'info' | 'success' | 'error' | 'challenge'

export interface Toast {
  id: string
  type: ToastType
  title: string
  message?: string
  duration?: number // ms, 0 = persistent
  action?: {
    label: string
    onClick: () => void
  }
  secondaryAction?: {
    label: string
    onClick: () => void
  }
  data?: Record<string, unknown>
}

interface ToastContextType {
  toasts: Toast[]
  showToast: (toast: Omit<Toast, 'id'>) => string
  dismissToast: (id: string) => void
  dismissAll: () => void
}

const ToastContext = createContext<ToastContextType | undefined>(undefined)

let toastIdCounter = 0
function generateToastId(): string {
  return `toast-${Date.now()}-${++toastIdCounter}`
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const showToast = useCallback(
    (toast: Omit<Toast, 'id'>): string => {
      const id = generateToastId()
      const newToast: Toast = { ...toast, id }

      setToasts((prev) => [...prev, newToast])

      // Auto-dismiss if duration is set (default 5s, 0 = persistent)
      const duration = toast.duration ?? (toast.type === 'challenge' ? 0 : 5000)
      if (duration > 0) {
        setTimeout(() => {
          dismissToast(id)
        }, duration)
      }

      return id
    },
    [dismissToast]
  )

  const dismissAll = useCallback(() => {
    setToasts([])
  }, [])

  return (
    <ToastContext.Provider value={{ toasts, showToast, dismissToast, dismissAll }}>
      {children}
    </ToastContext.Provider>
  )
}

export function useToast() {
  const context = useContext(ToastContext)
  if (context === undefined) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return context
}
