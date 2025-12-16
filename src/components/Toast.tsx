/**
 * Toast notification component
 *
 * Renders stacked toast notifications in the bottom-right corner.
 * Supports info, success, error, and challenge variants.
 * Challenge toasts persist until user action.
 */

import { useToast, type Toast as ToastType, type ToastType as ToastVariant } from '../contexts/ToastContext'
import { Button } from './ui/button'

const toastStyles: Record<ToastVariant, { bg: string; border: string; icon: string }> = {
  info: {
    bg: 'bg-blue-50 dark:bg-blue-950',
    border: 'border-blue-200 dark:border-blue-800',
    icon: 'text-blue-600 dark:text-blue-400',
  },
  success: {
    bg: 'bg-green-50 dark:bg-green-950',
    border: 'border-green-200 dark:border-green-800',
    icon: 'text-green-600 dark:text-green-400',
  },
  error: {
    bg: 'bg-red-50 dark:bg-red-950',
    border: 'border-red-200 dark:border-red-800',
    icon: 'text-red-600 dark:text-red-400',
  },
  challenge: {
    bg: 'bg-amber-50 dark:bg-amber-950',
    border: 'border-amber-200 dark:border-amber-800',
    icon: 'text-amber-600 dark:text-amber-400',
  },
}

const icons: Record<ToastVariant, React.ReactNode> = {
  info: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  ),
  success: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  error: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
    </svg>
  ),
  challenge: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
}

function ToastItem({ toast }: { toast: ToastType }) {
  const { dismissToast } = useToast()
  const styles = toastStyles[toast.type]

  return (
    <div
      className={`
        ${styles.bg} ${styles.border}
        border rounded-lg shadow-lg p-4 min-w-[300px] max-w-[400px]
        animate-in slide-in-from-right-full duration-300
      `}
      role="alert"
    >
      <div className="flex gap-3">
        {/* Icon */}
        <div className={`flex-shrink-0 ${styles.icon}`}>
          {icons[toast.type]}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <p className="font-medium text-sm text-foreground">{toast.title}</p>
          {toast.message && (
            <p className="mt-1 text-sm text-muted-foreground">{toast.message}</p>
          )}

          {/* Actions */}
          {(toast.action || toast.secondaryAction) && (
            <div className="mt-3 flex gap-2">
              {toast.action && (
                <Button
                  size="sm"
                  onClick={() => {
                    toast.action?.onClick()
                    dismissToast(toast.id)
                  }}
                >
                  {toast.action.label}
                </Button>
              )}
              {toast.secondaryAction && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    toast.secondaryAction?.onClick()
                    dismissToast(toast.id)
                  }}
                >
                  {toast.secondaryAction.label}
                </Button>
              )}
            </div>
          )}
        </div>

        {/* Dismiss button */}
        <button
          onClick={() => dismissToast(toast.id)}
          className="flex-shrink-0 text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Dismiss"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  )
}

export default function ToastContainer() {
  const { toasts } = useToast()

  if (toasts.length === 0) return null

  return (
    <div
      className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2"
      aria-live="polite"
      aria-label="Notifications"
    >
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} />
      ))}
    </div>
  )
}
