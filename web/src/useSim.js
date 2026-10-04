import { useEffect, useRef, useState } from 'react'

// The server sends a frame every 100 ms. We keep the last two frames so the
// cars can be animated smoothly in between (see Cars.jsx).
export const FRAME_MS = 100
const HISTORY_LEN = 120

function wsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}/ws`
}

export function useSim() {
  const [layout, setLayout] = useState(null)
  const [stats, setStats] = useState(null)
  const [status, setStatus] = useState('connecting')
  const [controls, setControls] = useState({ paused: false, speed: 1 })
  // wait-time chart points: [sim time, avg wait last minute, light mode]
  const [history, setHistory] = useState([])
  const [aiAvailable, setAiAvailable] = useState(true)

  // frames live in a ref, not state, so 10 updates a second don't re-render React
  const frames = useRef({ prev: null, curr: null, receivedAt: 0 })

  useEffect(() => {
    let ws
    let retry
    let closed = false

    function connect() {
      ws = new WebSocket(wsUrl())
      ws.onopen = () => setStatus('live')
      ws.onclose = () => {
        setStatus('offline')
        if (!closed) retry = setTimeout(connect, 2000)
      }
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === 'layout') {
          setLayout(msg)
          setHistory(msg.history || [])
          setAiAvailable(msg.ai_available !== false)
          frames.current = { prev: null, curr: null, receivedAt: 0 }
          return
        }
        const f = frames.current
        f.prev = f.curr
        f.curr = msg
        f.receivedAt = performance.now()
        setStats(msg.stats)
        if (msg.samples && msg.samples.length) {
          setHistory((h) => {
            const lastT = h.length ? h[h.length - 1][0] : -1
            // a sample from earlier than our last one means the city was reset
            if (msg.samples[0][0] < lastT) return msg.samples
            const fresh = msg.samples.filter((p) => p[0] > lastT)
            return fresh.length ? [...h, ...fresh].slice(-HISTORY_LEN) : h
          })
        }
        setControls((c) =>
          c.paused === msg.paused && c.speed === msg.speed ? c : { paused: msg.paused, speed: msg.speed }
        )
      }
    }

    connect()
    return () => {
      closed = true
      clearTimeout(retry)
      ws && ws.close()
    }
  }, [])

  return { layout, stats, status, controls, frames, history, aiAvailable }
}

export async function post(path, body) {
  const res = await fetch(`/api/${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)
  return data
}
