import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ingest': 'http://localhost:8007',
      '/jobs': 'http://localhost:8007',
      '/incidents': 'http://localhost:8007',
      '/device-profiles': 'http://localhost:8007',
      '/health': 'http://localhost:8007',
    },
  },
})
