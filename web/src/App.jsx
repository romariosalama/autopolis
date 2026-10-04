import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { useState } from 'react'
import CityMap from './CityMap'
import Cars from './Cars'
import Lights from './Lights'
import WaitChart from './WaitChart'
import { post, useSim } from './useSim'

const SPEEDS = [1, 2, 5, 10]

function formatClock(seconds) {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${String(s).padStart(2, '0')}`
}

function StatsSign({ stats, status }) {
  return (
    <section className="sign" aria-label="City stats">
      <div className="sign-inner">
        <h1>Autopolis</h1>
        {stats ? (
          <dl>
            <div><dt>Avg wait, last min</dt><dd>{stats.avg_wait_recent.toFixed(1)}<small>s</small></dd></div>
            <div><dt>Cars per minute</dt><dd>{stats.cars_per_minute.toFixed(0)}</dd></div>
            <div><dt>Trips completed</dt><dd>{stats.trips_completed.toLocaleString()}</dd></div>
            <div><dt>Cars on the road</dt><dd>{stats.cars}</dd></div>
          </dl>
        ) : (
          <p className="sign-note">{status === 'offline' ? 'Can’t reach the simulation server. Retrying…' : 'Connecting to the city…'}</p>
        )}
        {stats && (
          <p className="sign-foot">
            <span className={`dot ${status}`} aria-hidden="true" />
            {status === 'live' ? 'Live' : 'Reconnecting'} · city time {formatClock(stats.time)}
          </p>
        )}
      </div>
    </section>
  )
}

function Controls({ stats, controls }) {
  const [message, setMessage] = useState(null)

  async function run(path, body) {
    try {
      setMessage(null)
      await post(path, body)
    } catch (e) {
      setMessage(e.message === 'no trained model loaded yet'
        ? 'AI lights aren’t trained yet, so the city is still on fixed timers.'
        : e.message)
    }
  }

  const mode = stats?.light_mode ?? 'fixed'
  const rush = stats?.rush_hour ?? false

  return (
    <section className="controls" aria-label="Simulation controls">
      <div className="group">
        <button onClick={() => run(controls.paused ? 'resume' : 'pause')}>
          {controls.paused ? 'Resume' : 'Pause'}
        </button>
      </div>

      <div className="group" role="radiogroup" aria-label="Simulation speed">
        <span className="label">Speed</span>
        {SPEEDS.map((s) => (
          <button key={s} role="radio" aria-checked={controls.speed === s}
            className={controls.speed === s ? 'on' : ''}
            onClick={() => run('speed', { speed: s })}>
            {s}×
          </button>
        ))}
      </div>

      <div className="group">
        <button onClick={() => run('spawn', { count: 25 })}>Add 25 cars</button>
        <button aria-pressed={rush} className={rush ? 'on' : ''}
          onClick={() => run('rush-hour', { on: !rush })}>
          Rush hour
        </button>
      </div>

      <div className="group" role="radiogroup" aria-label="Traffic lights">
        <span className="label">Lights</span>
        <button role="radio" aria-checked={mode === 'fixed'} className={mode === 'fixed' ? 'on' : ''}
          onClick={() => run('lights', { mode: 'fixed' })}>
          Fixed timer
        </button>
        <button role="radio" aria-checked={mode === 'ai'} className={mode === 'ai' ? 'on' : ''}
          onClick={() => run('lights', { mode: 'ai' })}>
          AI
        </button>
      </div>

      {message && <p className="message" role="status">{message}</p>}
    </section>
  )
}

export default function App() {
  const { layout, stats, status, controls, frames, history, aiAvailable } = useSim()

  const center = layout
    ? [((layout.cols - 1) * layout.block) / 2, 0, ((layout.rows - 1) * layout.block) / 2]
    : [0, 0, 0]
  const span = layout ? Math.max(layout.cols, layout.rows) * layout.block : 400

  return (
    <main>
      <Canvas
        shadows
        camera={{ position: [center[0] - span * 0.22, span * 0.3, center[2] + span * 0.45], fov: 45, near: 1, far: 8000 }}
        key={layout ? `${layout.rows}x${layout.cols}` : 'empty'}
      >
        <color attach="background" args={['#b9c4c9']} />
        <fog attach="fog" args={['#b9c4c9', span * 0.9, span * 2.2]} />
        <hemisphereLight args={['#e8eef2', '#4a4a44', 0.9]} />
        <directionalLight
          position={[center[0] + 250, 400, center[2] - 150]}
          intensity={1.6}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-span}
          shadow-camera-right={span}
          shadow-camera-top={span}
          shadow-camera-bottom={-span}
          shadow-camera-far={1500}
        />
        {layout && (
          <>
            <CityMap layout={layout} />
            <Lights layout={layout} frames={frames} />
            <Cars frames={frames} />
          </>
        )}
        <OrbitControls target={center} maxPolarAngle={Math.PI / 2.2} minDistance={40} maxDistance={span * 2.5} />
      </Canvas>

      <div className="side">
        <StatsSign stats={stats} status={status} />
        <WaitChart history={history} aiAvailable={aiAvailable} />
      </div>
      <Controls stats={stats} controls={controls} />
    </main>
  )
}
