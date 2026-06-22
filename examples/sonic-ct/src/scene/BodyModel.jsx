import React, { Suspense, useMemo } from "react";
import * as THREE from "three";
import { useGLTF } from "@react-three/drei";
import { theme } from "../theme.js";
import { TORSO_PROFILE, SX, SZ } from "./anatomy.js";
import manifest from "../organ_manifest.json";

// Set to a path under /public (e.g. "/models/torso.glb") to use a supplied
// anatomy model. Empty string => procedural ghost fallback. The GLB is a
// VISUAL prior only; the Rust phantom remains the physics ground truth.
export const GLB_URL = "";

// ---------------------------------------------------------------------------
// Ghost material override + organ colouring (applied to a loaded GLB)
// ---------------------------------------------------------------------------

function ghostMaterial(color) {
  const g = manifest.ghostMaterial || {};
  return new THREE.MeshPhysicalMaterial({
    color: color || g.color || theme.cyan,
    emissive: new THREE.Color(g.emissive || theme.cyan),
    emissiveIntensity: g.emissiveIntensity ?? 0.28,
    transparent: true,
    opacity: g.opacity ?? 0.22,
    roughness: 0.18,
    metalness: 0.0,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
}

function organColorFor(name) {
  const lname = name.toLowerCase();
  for (const spec of Object.values(manifest.organs || {})) {
    if (spec.names.some((n) => lname.includes(n.toLowerCase()))) return spec.color;
  }
  return null;
}

export function applyGhostMaterials(root) {
  root.traverse((obj) => {
    if (obj.isMesh) {
      obj.material = ghostMaterial(organColorFor(obj.name));
      obj.material.blending = THREE.AdditiveBlending;
      obj.renderOrder = 10;
    }
  });
}

function LoadedGlbBody({ url }) {
  const gltf = useGLTF(url);
  const scene = useMemo(() => {
    const s = gltf.scene.clone(true);
    applyGhostMaterials(s);
    return s;
  }, [gltf]);
  return <primitive object={scene} />;
}

// ---------------------------------------------------------------------------
// Procedural ghost fallback (runs with no licensed assets)
// ---------------------------------------------------------------------------

export function ProceduralGhost({ bodyH }) {
  const geom = useMemo(() => {
    const pts = TORSO_PROFILE.map(([t, r]) => new THREE.Vector2(r, (t - 0.5) * bodyH));
    return new THREE.LatheGeometry(pts, 72);
  }, [bodyH]);

  const y = (t) => (t - 0.5) * bodyH;
  const organs = [
    { p: [-0.04, y(0.84), 0.06], r: 0.15, c: theme.danger, n: "heart" },
    { p: [-0.32, y(0.86), -0.02], r: 0.18, c: theme.blue, n: "lungL" },
    { p: [0.32, y(0.86), -0.02], r: 0.18, c: theme.blue, n: "lungR" },
    { p: [0.24, y(0.68), 0.04], r: 0.21, c: theme.amber, n: "liver" },
    { p: [-0.26, y(0.64), -0.04], r: 0.12, c: theme.violet, n: "spleen" },
    { p: [0.18, y(0.48), -0.12], r: 0.11, c: theme.success, n: "kidneyR" },
    { p: [-0.18, y(0.48), -0.12], r: 0.11, c: theme.success, n: "kidneyL" },
    { p: [0.0, y(0.28), 0.02], r: 0.22, c: theme.violet, n: "bowel" },
  ];

  return (
    <group scale={[SX, 1, SZ]}>
      {/* violet volumetric glow, brightest at the silhouette */}
      <mesh geometry={geom}>
        <meshBasicMaterial color={theme.violet} transparent opacity={0.3} side={THREE.BackSide} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <mesh geometry={geom} scale={[0.88, 1, 0.88]}>
        <meshBasicMaterial color={theme.blue} transparent opacity={0.18} side={THREE.BackSide} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      <mesh geometry={geom} scale={[0.7, 1, 0.7]}>
        <meshBasicMaterial color={theme.violet} transparent opacity={0.12} side={THREE.BackSide} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      {/* spine + internal organ glows */}
      <mesh position={[0, 0, -0.42]}>
        <cylinderGeometry args={[0.05, 0.06, bodyH * 0.95, 12]} />
        <meshBasicMaterial color={theme.medicalWhite} transparent opacity={0.3} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
      </mesh>
      {organs.map((o) => (
        <mesh key={o.n} position={o.p} scale={[1, 1.25, 0.8]}>
          <sphereGeometry args={[o.r, 20, 20]} />
          <meshBasicMaterial color={o.c} transparent opacity={0.42} blending={THREE.AdditiveBlending} depthWrite={false} toneMapped={false} />
        </mesh>
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Error boundary so a missing/broken GLB cleanly falls back to procedural
// ---------------------------------------------------------------------------

class GlbBoundary extends React.Component {
  constructor(p) {
    super(p);
    this.state = { failed: false };
  }
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}

export default function BodyModel({ bodyH }) {
  const fallback = <ProceduralGhost bodyH={bodyH} />;
  if (!GLB_URL) return fallback;
  return (
    <GlbBoundary fallback={fallback}>
      <Suspense fallback={fallback}>
        <LoadedGlbBody url={GLB_URL} />
      </Suspense>
    </GlbBoundary>
  );
}
