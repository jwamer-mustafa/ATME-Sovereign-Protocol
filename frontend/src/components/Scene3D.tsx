import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment } from '@react-three/drei';
import * as THREE from 'three';

interface SimState {
  environment?: {
    agent?: {
      position: number[];
      orientation: number;
      velocity: number[];
    };
    objects?: Array<{
      id: number;
      position: number[];
      orientation: number[];
    }>;
    injected_entities?: Array<{
      type: string;
      position: number[];
    }>;
  };
}

function Agent({ state }: { state: SimState | null }) {
  const meshRef = useRef<THREE.Mesh>(null);
  const arrowRef = useRef<THREE.Mesh>(null);

  useFrame(() => {
    if (!meshRef.current || !state?.environment?.agent) return;
    const { position, orientation } = state.environment.agent;
    meshRef.current.position.set(position[0], position[2], -position[1]);
    meshRef.current.rotation.set(0, -orientation, 0);
    if (arrowRef.current) {
      arrowRef.current.position.set(position[0], position[2] + 0.8, -position[1]);
      arrowRef.current.rotation.set(0, -orientation, 0);
    }
  });

  return (
    <>
      <mesh ref={meshRef} castShadow>
        <capsuleGeometry args={[0.3, 1.0, 8, 16]} />
        <meshStandardMaterial color="#1aff5e" emissive="#0a8030" emissiveIntensity={0.3} />
      </mesh>
      <mesh ref={arrowRef}>
        <coneGeometry args={[0.15, 0.4, 8]} />
        <meshStandardMaterial color="#ffff00" emissive="#888800" emissiveIntensity={0.5} />
      </mesh>
    </>
  );
}

function WorldObjects({ state }: { state: SimState | null }) {
  const colors = ['#ff4444', '#4444ff', '#ffff44', '#ff44ff', '#44ffff'];

  if (!state?.environment?.objects) return null;

  return (
    <>
      {state.environment.objects.map((obj, i) => (
        <mesh
          key={obj.id}
          position={[obj.position[0], obj.position[2], -obj.position[1]]}
          castShadow
        >
          <boxGeometry args={[0.5, 0.5, 0.5]} />
          <meshStandardMaterial
            color={colors[i % colors.length]}
            emissive={colors[i % colors.length]}
            emissiveIntensity={0.1}
          />
        </mesh>
      ))}
    </>
  );
}

function InjectedEntities({ state }: { state: SimState | null }) {
  const entities = state?.environment?.injected_entities;
  if (!entities?.length) return null;

  return (
    <>
      {entities.map((entity, i) => (
        <mesh
          key={i}
          position={[entity.position[0], entity.position[2] || 1, -entity.position[1]]}
        >
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color={entity.type === 'query_orb' ? '#3399ff' : '#ffcc00'}
            emissive={entity.type === 'query_orb' ? '#1155aa' : '#886600'}
            emissiveIntensity={0.5}
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}
    </>
  );
}

function Arena() {
  const wallMaterial = useMemo(
    () => new THREE.MeshStandardMaterial({ color: '#444466', transparent: true, opacity: 0.3 }),
    []
  );

  return (
    <>
      <Grid
        args={[20, 20]}
        cellSize={1}
        cellColor="#222244"
        sectionSize={5}
        sectionColor="#333366"
        fadeDistance={25}
        fadeStrength={1}
        position={[0, 0, 0]}
      />
      {/* Walls */}
      <mesh position={[0, 1, -5]} material={wallMaterial}>
        <boxGeometry args={[10, 2, 0.2]} />
      </mesh>
      <mesh position={[0, 1, 5]} material={wallMaterial}>
        <boxGeometry args={[10, 2, 0.2]} />
      </mesh>
      <mesh position={[-5, 1, 0]} material={wallMaterial}>
        <boxGeometry args={[0.2, 2, 10]} />
      </mesh>
      <mesh position={[5, 1, 0]} material={wallMaterial}>
        <boxGeometry args={[0.2, 2, 10]} />
      </mesh>
      {/* Target spheres */}
      <mesh position={[3, 0.5, -2]}>
        <sphereGeometry args={[0.4, 16, 16]} />
        <meshStandardMaterial color="#00ff00" emissive="#008800" emissiveIntensity={0.3} transparent opacity={0.7} />
      </mesh>
      <mesh position={[-2, 0.5, 3]}>
        <sphereGeometry args={[0.4, 16, 16]} />
        <meshStandardMaterial color="#00ff00" emissive="#008800" emissiveIntensity={0.3} transparent opacity={0.7} />
      </mesh>
    </>
  );
}

export function Scene3D({ state }: { state: SimState | null }) {
  return (
    <Canvas
      camera={{ position: [12, 10, 12], fov: 50 }}
      shadows
      style={{ background: '#0a0a1a' }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[10, 15, 10]} intensity={0.8} castShadow />
      <pointLight position={[0, 5, 0]} intensity={0.3} color="#6666ff" />

      <Arena />
      <Agent state={state} />
      <WorldObjects state={state} />
      <InjectedEntities state={state} />

      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        maxPolarAngle={Math.PI / 2.1}
        minDistance={3}
        maxDistance={25}
      />
    </Canvas>
  );
}
