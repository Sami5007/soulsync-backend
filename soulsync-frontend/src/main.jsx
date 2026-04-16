import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

// ADDED: Import the service worker registration from the Vite PWA plugin
import { registerSW } from 'virtual:pwa-register'

// ADDED: Start the service worker immediately
registerSW({ immediate: true })

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)