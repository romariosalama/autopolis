import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// in dev, forward the API and WebSocket to the FastAPI server
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
})
