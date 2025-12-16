declare const __COMMIT_HASH__: string
declare const __BUILD_DATE__: string

export default function Footer() {
  const buildDate = new Date(__BUILD_DATE__)
  const formattedDate = buildDate.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })

  return (
    <footer className="py-4 text-center text-xs text-muted-foreground/60">
      <span>
        v{__COMMIT_HASH__} &middot; Updated {formattedDate}
      </span>
    </footer>
  )
}
