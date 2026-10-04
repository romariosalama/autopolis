import { useMemo, useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import { FRAME_MS } from './useSim'

// One InstancedMesh draws every car in a single draw call, which is what
// keeps 500+ cars smooth in the browser.
const MAX_CARS = 2000
const CAR_COLORS = ['#d8d4cc', '#2b2f36', '#7d8590', '#a33a2c', '#2f4f74', '#c9a227', '#4b6a4f', '#f0efea']

function lerpAngle(a, b, t) {
  let d = b - a
  while (d > Math.PI) d -= 2 * Math.PI
  while (d < -Math.PI) d += 2 * Math.PI
  return a + d * t
}

export default function Cars({ frames }) {
  const mesh = useRef()
  const dummy = useMemo(() => new THREE.Object3D(), [])
  const color = useMemo(() => new THREE.Color(), [])

  useFrame(() => {
    const { prev, curr, receivedAt } = frames.current
    if (!mesh.current || !curr) return

    // how far we are between the previous frame and the current one
    const t = Math.min((performance.now() - receivedAt) / FRAME_MS, 1)
    const prevById = new Map()
    if (prev) for (const c of prev.cars) prevById.set(c[0], c)

    const cars = curr.cars
    const count = Math.min(cars.length, MAX_CARS)
    for (let i = 0; i < count; i++) {
      const [id, x, y, heading] = cars[i]
      const p = prevById.get(id)
      let px = x, py = y, ph = heading
      // only interpolate short moves; a big jump means the car turned a corner
      if (p && Math.hypot(p[1] - x, p[2] - y) < 15) {
        px = p[1] + (x - p[1]) * t
        py = p[2] + (y - p[2]) * t
        ph = lerpAngle(p[3], heading, t)
      }
      dummy.position.set(px, 0.85, py)
      dummy.rotation.set(0, -ph, 0)
      dummy.updateMatrix()
      mesh.current.setMatrixAt(i, dummy.matrix)
      mesh.current.setColorAt(i, color.set(CAR_COLORS[id % CAR_COLORS.length]))
    }
    mesh.current.count = count
    mesh.current.instanceMatrix.needsUpdate = true
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true
  })

  return (
    <instancedMesh ref={mesh} args={[null, null, MAX_CARS]} castShadow frustumCulled={false}>
      <boxGeometry args={[4.4, 1.5, 2.0]} />
      <meshStandardMaterial roughness={0.5} metalness={0.2} />
    </instancedMesh>
  )
}
