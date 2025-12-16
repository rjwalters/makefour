import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

declare const __COMMIT_HASH__: string
declare const __BUILD_DATE__: string

console.log(`MakeFour v${__COMMIT_HASH__} | Built ${__BUILD_DATE__}`)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
