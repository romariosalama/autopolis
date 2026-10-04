import { useMemo, useState } from 'react'

// Average wait over the last 10 minutes of city time. Stretches where the
// AI was running the lights get a green band, so flipping the toggle and
// watching the line drop is the whole point of this chart.

const W = 300
const H = 124
const PAD = { top: 8, right: 8, bottom: 20, left: 32 }

function niceMax(v) {
  if (v <= 10) return 10
  const step = v <= 50 ? 10 : 20
  return Math.ceil(v / step) * step
}

function clock(t) {
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

export default function WaitChart({ history, aiAvailable }) {
  const [hover, setHover] = useState(null)

  const geom = useMemo(() => {
    if (history.length < 2) return null
    const t0 = history[0][0]
    const t1 = Math.max(history[history.length - 1][0], t0 + 60)
    const yMax = niceMax(Math.max(...history.map((p) => p[1])))
    const iw = W - PAD.left - PAD.right
    const ih = H - PAD.top - PAD.bottom
    const x = (t) => PAD.left + ((t - t0) / (t1 - t0)) * iw
    const y = (v) => PAD.top + ih - (v / yMax) * ih

    const line = history.map((p, i) => `${i ? 'L' : 'M'}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join('')

    // contiguous runs of AI samples become shaded bands
    const bands = []
    let start = null
    history.forEach((p, i) => {
      if (p[2] === 'ai' && start === null) start = i
      const endOfRun = p[2] !== 'ai' || i === history.length - 1
      if (start !== null && endOfRun) {
        const end = p[2] === 'ai' ? i : i - 1
        const left = start > 0 ? (history[start - 1][0] + history[start][0]) / 2 : history[start][0]
        bands.push({ x0: x(left), x1: x(history[end][0]) })
        start = null
      }
    })

    return { x, y, yMax, line, bands, t0, t1, ih }
  }, [history])

  const hasAi = history.some((p) => p[2] === 'ai')

  function onMove(e) {
    if (!geom) return
    const rect = e.currentTarget.getBoundingClientRect()
    const px = ((e.clientX - rect.left) / rect.width) * W
    let best = null
    for (const p of history) {
      const d = Math.abs(geom.x(p[0]) - px)
      if (!best || d < best.d) best = { d, p }
    }
    setHover(best && best.p)
  }

  return (
    <section className="chart-panel" aria-label="Average wait over time">
      <h2>Average wait over time</h2>
      {geom ? (
        <div className="chart-wrap">
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMinYMid meet" onMouseMove={onMove} onMouseLeave={() => setHover(null)}
            role="img" aria-label={`Average wait, currently ${history[history.length - 1][1].toFixed(1)} seconds`}>
            {geom.bands.map((b, i) => (
              <rect key={i} className="ai-band" x={b.x0} y={PAD.top} width={Math.max(b.x1 - b.x0, 2)} height={geom.ih} />
            ))}
            {[0, geom.yMax / 2, geom.yMax].map((v) => (
              <g key={v}>
                <line className="grid" x1={PAD.left} x2={W - PAD.right} y1={geom.y(v)} y2={geom.y(v)} />
                <text className="tick" x={PAD.left - 6} y={geom.y(v) + 3} textAnchor="end">{v}s</text>
              </g>
            ))}
            <text className="tick" x={PAD.left} y={H - 4}>{clock(geom.t0)}</text>
            <text className="tick" x={W - PAD.right} y={H - 4} textAnchor="end">{clock(history[history.length - 1][0])}</text>
            <path className="wait-line" d={geom.line} />
            {hover && (
              <g>
                <line className="crosshair" x1={geom.x(hover[0])} x2={geom.x(hover[0])} y1={PAD.top} y2={PAD.top + geom.ih} />
                <circle className="hover-dot" cx={geom.x(hover[0])} cy={geom.y(hover[1])} r="4" />
              </g>
            )}
          </svg>
        </div>
      ) : (
        <p className="chart-empty">Collecting data…</p>
      )}
      <p className="chart-note" aria-live="polite">
        {hover ? (
          <span className="readout">
            At {clock(hover[0])}: {hover[1].toFixed(1)}s wait with {hover[2] === 'ai' ? 'AI lights' : 'fixed timers'}
          </span>
        ) : hasAi ? (
          <><span className="swatch" aria-hidden="true" />AI lights on</>
        ) : aiAvailable ? (
          'Switch the lights to AI and watch this line.'
        ) : (
          'AI lights aren’t loaded on this server.'
        )}
      </p>
    </section>
  )
}
