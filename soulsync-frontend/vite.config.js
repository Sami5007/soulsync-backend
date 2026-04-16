import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['Logo-soulsync.png'], // Make sure this exact file is in your public folder
      manifest: {
        short_name: 'Soul-Sync',
        name: 'Soul-Sync: Mental Wellness Companion',
        description: 'Your compassionate companion for mental wellness',
        theme_color: '#1976D2',
        background_color: '#F5F5F5',
        display: 'standalone',
        orientation: 'portrait',
        icons: [
          {
            src: '/Logo-soulsync.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable'
          },
          {
            src: '/Logo-soulsync.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      }
    })
  ],
});