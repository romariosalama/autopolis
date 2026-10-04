import { useMemo } from 'react'

// Static parts of the city: ground, roads, sidewalks, and buildings.
// Sim coordinates are (x, y) on a flat plane; in three.js that's (x, z).

const ROAD_WIDTH = 11
const COLORS = {
  ground: '#59614f',
  asphalt: '#2d3034',
  intersection: '#33363b',
  sidewalk: '#8d8c86',
  paint: '#e9e5d6',
  buildings: ['#ddd6c6', '#c4bdae', '#cfd1cb', '#a9afb5', '#e3dac6', '#b8b3a6'],
}

// small seeded random so the buildings look the same every time
function seeded(seed) {
  let s = seed * 9301 + 49297
  return () => {
    s = (s * 9301 + 49297) % 233280
    return s / 233280
  }
}

function Road({ a, b }) {
  const dx = b.x - a.x
  const dz = b.y - a.y
  const length = Math.hypot(dx, dz)
  const horizontal = Math.abs(dx) > Math.abs(dz)
  const cx = (a.x + b.x) / 2
  const cz = (a.y + b.y) / 2

  // dashed center line
  const dashes = []
  for (let d = 10; d < length - 10; d += 8) {
    const t = d / length - 0.5
    dashes.push(
      <mesh key={d} position={[cx + dx * t, 0.12, cz + dz * t]} rotation={[-Math.PI / 2, 0, horizontal ? 0 : Math.PI / 2]}>
        <planeGeometry args={[3.5, 0.3]} />
        <meshStandardMaterial color={COLORS.paint} />
      </mesh>
    )
  }

  return (
    <group>
      <mesh position={[cx, 0.06, cz]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={horizontal ? [length, ROAD_WIDTH] : [ROAD_WIDTH, length]} />
        <meshStandardMaterial color={COLORS.asphalt} roughness={0.95} />
      </mesh>
      {dashes}
    </group>
  )
}

function Block({ x, z, size, seed }) {
  const buildings = useMemo(() => {
    const rand = seeded(seed)
    const list = []
    const inner = size - 6
    const cells = 3
    const cell = inner / cells
    for (let i = 0; i < cells; i++) {
      for (let j = 0; j < cells; j++) {
        if (rand() < 0.18) continue // leave some empty lots
        const w = cell * (0.6 + rand() * 0.3)
        const d = cell * (0.6 + rand() * 0.3)
        // mostly low-rise so the streets stay visible, with the odd tower
        const h = rand() < 0.12 ? 22 + rand() * 18 : 5 + rand() * 11
        list.push({
          x: x - inner / 2 + cell * (i + 0.5),
          z: z - inner / 2 + cell * (j + 0.5),
          w, d, h,
          color: COLORS.buildings[Math.floor(rand() * COLORS.buildings.length)],
        })
      }
    }
    return list
  }, [x, z, size, seed])

  return (
    <group>
      <mesh position={[x, 0.15, z]} receiveShadow>
        <boxGeometry args={[size, 0.3, size]} />
        <meshStandardMaterial color={COLORS.sidewalk} />
      </mesh>
      {buildings.map((b, i) => (
        <mesh key={i} position={[b.x, b.h / 2 + 0.3, b.z]} castShadow receiveShadow>
          <boxGeometry args={[b.w, b.h, b.d]} />
          <meshStandardMaterial color={b.color} roughness={0.85} />
        </mesh>
      ))}
    </group>
  )
}

export default function CityMap({ layout }) {
  const nodes = useMemo(() => Object.fromEntries(layout.nodes.map((n) => [n.id, n])), [layout])
  const L = layout.block
  const blockSize = L - ROAD_WIDTH

  const blocks = []
  for (let r = 0; r < layout.rows - 1; r++) {
    for (let c = 0; c < layout.cols - 1; c++) {
      blocks.push({ x: c * L + L / 2, z: r * L + L / 2, seed: r * 31 + c + 1 })
    }
  }

  const width = (layout.cols - 1) * L
  const depth = (layout.rows - 1) * L

  return (
    <group>
      <mesh position={[width / 2, -0.05, depth / 2]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
        <planeGeometry args={[width + 6000, depth + 6000]} />
        <meshStandardMaterial color={COLORS.ground} roughness={1} />
      </mesh>

      {layout.roads.map((r, i) => (
        <Road key={i} a={nodes[r.a]} b={nodes[r.b]} />
      ))}

      {layout.nodes.filter((n) => !n.gate).map((n) => (
        <mesh key={n.id} position={[n.x, 0.09, n.y]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[ROAD_WIDTH, ROAD_WIDTH]} />
          <meshStandardMaterial color={COLORS.intersection} roughness={0.95} />
        </mesh>
      ))}

      {blocks.map((b, i) => (
        <Block key={i} x={b.x} z={b.z} size={blockSize} seed={b.seed} />
      ))}
    </group>
  )
}
