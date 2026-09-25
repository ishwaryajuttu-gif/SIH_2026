import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// During development the dashboard runs on :5173 and forwards API calls to FastAPI on :8000.
const BACKEND = process.env.BAS_BACKEND ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': BACKEND,
      '/video_feed': BACKEND,
      '/ws': { target: BACKEND.replace('http', 'ws'), ws: true },
    },
  },
})
