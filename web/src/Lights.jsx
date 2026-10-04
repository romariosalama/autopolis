import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// A signal head on each corner of each intersection. The two heads facing
// north/south traffic show the NS color, the other two show the EW color.
const SIGNAL = { green: '#2fd27a', yellow: '#ffb81f', red: '#ff4433' }
const OFFSET = 7.5

function colorFor(phase, axis) {
  if (phase === `${axis}_GREEN`) return SIGNAL.green
  if (phase === `${axis}_YELLOW`) return SIGNAL.yellow
  return SIGNAL.red
}

function Signal({ x, z, axis, light }) {
  const bulb = useRef()
  const c = useMemo(() => new THREE.Color(), [])
  useFrame(() => {
    const phase = light.current
    if (!bulb.current || !phase) return
    c.set(colorFor(phase, axis))
    bulb.current.material.color.copy(c)
    bulb.current.material.emissive.copy(c)
  })
  return (
    <group position={[x, 0, z]}>
      <mesh position={[0, 2.5, 0]}>
        <cylinderGeometry args={[0.15, 0.15, 5]} />
        <meshStandardMaterial color="#1d1f22" />
      </mesh>
      <mesh ref={bulb} position={[0, 5.3, 0]}>
        <sphereGeometry args={[1.0, 16, 12]} />
        <meshStandardMaterial emissiveIntensity={2.2} toneMapped={false} />
      </mesh>
    </group>
  )
}

function Intersection({ node, frames }) {
  // ref holding this intersection's current phase, read every animation frame
  const light = useRef(null)
  useFrame(() => {
    const curr = frames.current.curr
    if (!curr) return
    const l = curr.lights.find((l) => l.id === node.id)
    if (l) light.current = l.phase
  })
  const { x, y } = node
  return (
    <group>
      {/* corners: NS signals sit beside the north/south approaches */}
      <Signal x={x - OFFSET} z={y - OFFSET} axis="NS" light={light} />
      <Signal x={x + OFFSET} z={y + OFFSET} axis="NS" light={light} />
      <Signal x={x + OFFSET} z={y - OFFSET} axis="EW" light={light} />
      <Signal x={x - OFFSET} z={y + OFFSET} axis="EW" light={light} />
    </group>
  )
}

export default function Lights({ layout, frames }) {
  return layout.nodes
    .filter((n) => !n.gate)
    .map((n) => <Intersection key={n.id} node={n} frames={frames} />)
}
