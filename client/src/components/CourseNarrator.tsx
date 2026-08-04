"use client";

import {
  Cylinder,
  OrbitControls as DreiOrbitControls,
  RoundedBox,
  Sphere,
} from "@react-three/drei";
import { Canvas, useFrame } from "@react-three/fiber";
import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export type Viseme = "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "X";

export interface VisemeCue {
  start: number;
  end: number;
  viseme: Viseme;
}

export interface CourseNarratorProps {
  audioUrl: string;
  visemeTimeline: VisemeCue[];
  animation?: string;
  autoPlay?: boolean;
  isPaused?: boolean;
  onComplete?: () => void;
}

// Rough open/round shape per Rhubarb viseme category, applied to the mouth
// plane's scale. Not full blendshape morphing — the mouth is a flat plane,
// not a rigged face — but distinct enough per-category to read as real sync
// rather than the old amplitude-only sine flap.
const VISEME_MOUTH_SHAPE: Record<Viseme, { scaleY: number; scaleX: number }> = {
  X: { scaleY: 0.05, scaleX: 1 },
  A: { scaleY: 0.08, scaleX: 1 },
  B: { scaleY: 0.18, scaleX: 1 },
  C: { scaleY: 0.32, scaleX: 1.05 },
  D: { scaleY: 0.5, scaleX: 1.1 },
  E: { scaleY: 0.3, scaleX: 0.85 },
  F: { scaleY: 0.15, scaleX: 0.6 },
  G: { scaleY: 0.1, scaleX: 1 },
  H: { scaleY: 0.2, scaleX: 0.95 },
};

// While speaking, rotate through a few gesture animations instead of
// holding one static pose for the whole narration — keeps the hands
// visibly active. Picked purely off audio elapsed time inside useFrame,
// no React state, so it doesn't cost a re-render.
const SPEAKING_GESTURE_POOL = [
  "Teacher_ExplainingGestures",
  "Teacher_Emphasize",
  "Teacher_Talking",
  "Teacher_PointingBoard",
];
const GESTURE_ROTATE_SECONDS = 4;

function findActiveViseme(timeline: VisemeCue[], t: number): VisemeCue | null {
  for (const cue of timeline) {
    if (t >= cue.start && t < cue.end) return cue;
  }
  return null;
}

// --- Procedural Robot Component ---

interface RobotProps {
  currentAnimation: string;
  speaking: boolean;
  audioRef: React.RefObject<HTMLAudioElement | null>;
  visemeTimeline: VisemeCue[];
}

function RobotModel({
  currentAnimation,
  speaking,
  audioRef,
  visemeTimeline,
}: RobotProps) {
  const group = useRef<THREE.Group>(null);
  const headRef = useRef<THREE.Group>(null);
  const leftArmRef = useRef<THREE.Group>(null);
  const rightArmRef = useRef<THREE.Group>(null);
  const eyesRef = useRef<THREE.Group>(null);
  const leftEyeRef = useRef<THREE.Mesh>(null);
  const rightEyeRef = useRef<THREE.Mesh>(null);
  const mouthRef = useRef<THREE.Mesh>(null);

  const eyeColor = useRef(new THREE.Color("#00ffff"));
  const nextBlinkTime = useRef(0);

  useFrame((state, delta) => {
    const time = state.clock.getElapsedTime();
    const t = time * 2;

    let tHeadRot = 0;
    let tBodyRot = 0;
    let tLeftArm = [0, 0, 0.3];
    let tRightArm = [0, 0, -0.3];
    let tEyeColor = "#00ffff";
    let tEyeScale = [1, 1, 1];

    let blinkScale = 1;
    if (time > nextBlinkTime.current) {
      const blinkPhase = (time - nextBlinkTime.current) * 10;
      if (blinkPhase < Math.PI) {
        blinkScale = Math.max(0.1, Math.abs(Math.cos(blinkPhase)));
      } else {
        nextBlinkTime.current = time + 2 + Math.random() * 4;
      }
    }

    let effectiveAnimation = currentAnimation;
    if (speaking && audioRef.current) {
      const gestureIndex =
        Math.floor(audioRef.current.currentTime / GESTURE_ROTATE_SECONDS) %
        SPEAKING_GESTURE_POOL.length;
      effectiveAnimation = SPEAKING_GESTURE_POOL[gestureIndex];
    }

    switch (effectiveAnimation) {
      case "Teacher_StandingPose":
        tLeftArm = [Math.sin(t) * 0.1, 0, 0.3];
        tRightArm = [Math.sin(t + 1) * 0.1, 0, -0.3];
        break;
      case "Teacher_ExplainingGestures":
      case "Teacher_Talking": {
        tEyeColor = "#00ffaa";
        tEyeScale = [1.1, 1.1, 1];
        const wave = Math.sin(t * 4) * 0.35;
        const waveZ = Math.cos(t * 2.5) * 0.2;
        tLeftArm = [-0.8 + wave, 0.5, 0.6 + waveZ];
        tRightArm = [-0.8 + wave, -0.5, -0.6 - waveZ];
        tBodyRot = Math.sin(t * 1.5) * 0.15;
        tHeadRot = Math.sin(t * 2) * 0.05;
        break;
      }
      case "Teacher_ThinkingPose":
        tEyeColor = "#ffaa00";
        tEyeScale = [1, 0.3, 1];
        tHeadRot = 0.3;
        tRightArm = [-2.2, -0.5, 0.8];
        tLeftArm = [0, 0, 0.3];
        break;
      case "Teacher_PointingBoard":
        tEyeColor = "#ff00ff";
        tBodyRot = -0.2;
        tRightArm = [-1.3, 0, -0.8];
        tHeadRot = -0.4;
        break;
      case "Teacher_PointScreen":
        tEyeColor = "#ff00ff";
        tBodyRot = 0.3;
        tHeadRot = 0.4;
        tLeftArm = [-1.3, 0, 0.8];
        tRightArm = [0, 0, -0.3];
        break;
      case "Teacher_SwipeNext":
        tEyeColor = "#00ffff";
        tEyeScale = [1.2, 1, 1];
        tRightArm = [
          -0.5,
          -0.5 + Math.sin(t * 5) * 0.5,
          -0.5 + Math.cos(t * 5) * 0.5,
        ];
        tHeadRot = -0.1;
        break;
      case "Teacher_Emphasize": {
        tEyeColor = "#ffaa00";
        tEyeScale = [1.2, 1.2, 1];
        const chop = Math.sin(t * 8) * 0.35;
        tLeftArm = [-1.0 + chop, 0.5, 0.4];
        tRightArm = [-1.0 + chop, -0.5, -0.4];
        tHeadRot = Math.sin(t * 8) * 0.1;
        tBodyRot = 0;
        break;
      }
      case "Teacher_Listening":
        tEyeColor = "#00ffff";
        tEyeScale = [1, 0.8, 1];
        tLeftArm = [-0.5, -0.5, 0.5];
        tRightArm = [-0.5, 0.5, -0.5];
        tHeadRot = Math.sin(t * 0.5) * 0.05;
        break;
      case "Teacher_EncouragingNod":
        tEyeColor = "#00ff00";
        tEyeScale = [1.1, 0.6, 1];
        tHeadRot = Math.sin(t * 10) * 0.1;
        tLeftArm = [0.5, 0, 0.5];
        tRightArm = [0.5, 0, -0.5];
        break;
      case "Teacher_LookingAround":
        tHeadRot = Math.sin(t) * 0.5;
        break;
    }

    if (speaking) {
      tHeadRot += Math.sin(time * 15) * 0.05;
    }

    let tMouthScaleY = VISEME_MOUTH_SHAPE.X.scaleY;
    let tMouthScaleX = VISEME_MOUTH_SHAPE.X.scaleX;
    if (audioRef.current && visemeTimeline.length > 0) {
      const activeCue = findActiveViseme(
        visemeTimeline,
        audioRef.current.currentTime,
      );
      const shape = activeCue
        ? VISEME_MOUTH_SHAPE[activeCue.viseme]
        : VISEME_MOUTH_SHAPE.X;
      tMouthScaleY = shape.scaleY;
      tMouthScaleX = shape.scaleX;
    }

    const lerpFactor = 5 * delta;

    if (group.current) {
      group.current.rotation.y = THREE.MathUtils.lerp(
        group.current.rotation.y,
        tBodyRot,
        lerpFactor,
      );
      group.current.position.y = -0.6 + Math.sin(time) * 0.1;
    }

    if (headRef.current) {
      headRef.current.rotation.y = THREE.MathUtils.lerp(
        headRef.current.rotation.y,
        tHeadRot,
        lerpFactor,
      );
      headRef.current.rotation.z = THREE.MathUtils.lerp(
        headRef.current.rotation.z,
        Math.sin(t * 0.5) * 0.05,
        lerpFactor,
      );
    }

    if (leftArmRef.current) {
      leftArmRef.current.rotation.x = THREE.MathUtils.lerp(
        leftArmRef.current.rotation.x,
        tLeftArm[0],
        lerpFactor,
      );
      leftArmRef.current.rotation.y = THREE.MathUtils.lerp(
        leftArmRef.current.rotation.y,
        tLeftArm[1],
        lerpFactor,
      );
      leftArmRef.current.rotation.z = THREE.MathUtils.lerp(
        leftArmRef.current.rotation.z,
        tLeftArm[2],
        lerpFactor,
      );
    }
    if (rightArmRef.current) {
      rightArmRef.current.rotation.x = THREE.MathUtils.lerp(
        rightArmRef.current.rotation.x,
        tRightArm[0],
        lerpFactor,
      );
      rightArmRef.current.rotation.y = THREE.MathUtils.lerp(
        rightArmRef.current.rotation.y,
        tRightArm[1],
        lerpFactor,
      );
      rightArmRef.current.rotation.z = THREE.MathUtils.lerp(
        rightArmRef.current.rotation.z,
        tRightArm[2],
        lerpFactor,
      );
    }

    eyeColor.current.lerp(new THREE.Color(tEyeColor), lerpFactor);
    if (eyesRef.current) {
      const finalScaleY = tEyeScale[1] * blinkScale;
      eyesRef.current.scale.set(
        THREE.MathUtils.lerp(eyesRef.current.scale.x, tEyeScale[0], lerpFactor),
        THREE.MathUtils.lerp(
          eyesRef.current.scale.y,
          finalScaleY,
          lerpFactor * 2,
        ),
        1,
      );
    }
    if (leftEyeRef.current && rightEyeRef.current) {
      const leftMat = leftEyeRef.current.material as THREE.MeshStandardMaterial;
      const rightMat = rightEyeRef.current
        .material as THREE.MeshStandardMaterial;
      leftMat.color = eyeColor.current;
      leftMat.emissive = eyeColor.current;
      rightMat.color = eyeColor.current;
      rightMat.emissive = eyeColor.current;
    }

    if (mouthRef.current) {
      mouthRef.current.scale.y = THREE.MathUtils.lerp(
        mouthRef.current.scale.y,
        tMouthScaleY,
        lerpFactor * 8,
      );
      mouthRef.current.scale.x = THREE.MathUtils.lerp(
        mouthRef.current.scale.x,
        tMouthScaleX,
        lerpFactor * 8,
      );
      const mouthMat = mouthRef.current.material as THREE.MeshStandardMaterial;
      mouthMat.color = eyeColor.current;
      mouthMat.emissive = eyeColor.current;
    }
  });

  return (
    <group ref={group}>
      {/* --- HEAD --- */}
      <group ref={headRef} position={[0, 1.4, 0]}>
        <RoundedBox args={[0.5, 0.4, 0.45]} radius={0.1} smoothness={4}>
          <meshStandardMaterial
            color="#ffffff"
            emissive="#222222"
            metalness={0.6}
            roughness={0.2}
          />
        </RoundedBox>
        <RoundedBox
          args={[0.42, 0.3, 0.05]}
          radius={0.05}
          position={[0, 0, 0.21]}
        >
          <meshStandardMaterial
            color="#222222"
            metalness={0.9}
            roughness={0.1}
          />
        </RoundedBox>
        <group ref={eyesRef} position={[0, 0, 0.24]}>
          <PlaneEye meshRef={leftEyeRef} position={[-0.1, 0, 0]} />
          <PlaneEye meshRef={rightEyeRef} position={[0.1, 0, 0]} />
        </group>
        <mesh ref={mouthRef} position={[0, -0.12, 0.24]}>
          <planeGeometry args={[0.15, 0.05]} />
          <meshStandardMaterial
            color="#00ffff"
            emissive="#00ffff"
            emissiveIntensity={1}
          />
        </mesh>
        <group position={[0, 0.3, 0]}>
          <Cylinder args={[0.02, 0.02, 0.2]} position={[0, 0, 0]}>
            <meshStandardMaterial color="#888" />
          </Cylinder>
          <Sphere args={[0.05]} position={[0, 0.1, 0]}>
            <meshStandardMaterial
              color="#00ffff"
              emissive="#00ffff"
              emissiveIntensity={2}
            />
          </Sphere>
        </group>
      </group>

      {/* --- NECK --- */}
      <Cylinder args={[0.05, 0.05, 0.2]} position={[0, 1.15, 0]}>
        <meshStandardMaterial
          color="#0ea5e9"
          emissive="#004488"
          emissiveIntensity={0.5}
        />
      </Cylinder>

      {/* --- BODY --- */}
      <group position={[0, 0.6, 0]}>
        <RoundedBox
          args={[0.6, 0.7, 0.4]}
          radius={0.2}
          smoothness={4}
          position={[0, 0.1, 0]}
        >
          <meshStandardMaterial
            color="#ffffff"
            emissive="#222222"
            metalness={0.5}
            roughness={0.2}
          />
        </RoundedBox>
        <Cylinder
          args={[0.15, 0.15, 0.05]}
          rotation={[Math.PI / 2, 0, 0]}
          position={[0, 0.1, 0.2]}
        >
          <meshStandardMaterial
            color="#00ffff"
            emissive="#00ffff"
            emissiveIntensity={1}
          />
        </Cylinder>
      </group>

      {/* --- ARMS --- */}
      <group position={[0.35, 0.9, 0]}>
        <group ref={leftArmRef}>
          <Sphere args={[0.12]} position={[0, 0, 0]}>
            <meshStandardMaterial
              color="#0ea5e9"
              metalness={0.8}
              roughness={0.2}
            />
          </Sphere>
          <RoundedBox args={[0.1, 0.4, 0.1]} position={[0, -0.25, 0]}>
            <meshStandardMaterial color="#ffffff" emissive="#111111" />
          </RoundedBox>
          <Sphere args={[0.1]} position={[0, -0.5, 0]}>
            <meshStandardMaterial
              color="#0ea5e9"
              metalness={0.8}
              roughness={0.2}
            />
          </Sphere>
        </group>
      </group>

      <group position={[-0.35, 0.9, 0]}>
        <group ref={rightArmRef}>
          <Sphere args={[0.12]} position={[0, 0, 0]}>
            <meshStandardMaterial
              color="#0ea5e9"
              metalness={0.8}
              roughness={0.2}
            />
          </Sphere>
          <RoundedBox args={[0.1, 0.4, 0.1]} position={[0, -0.25, 0]}>
            <meshStandardMaterial color="#ffffff" emissive="#111111" />
          </RoundedBox>
          <Sphere args={[0.1]} position={[0, -0.5, 0]}>
            <meshStandardMaterial
              color="#0ea5e9"
              metalness={0.8}
              roughness={0.2}
            />
          </Sphere>
        </group>
      </group>

      {/* --- HOVER ENGINE / BASE --- */}
      <group position={[0, 0.1, 0]}>
        <Cylinder args={[0.1, 0.2, 0.3]} position={[0, 0, 0]}>
          <meshStandardMaterial color="#334155" />
        </Cylinder>
        <pointLight
          color="#00ffff"
          intensity={1}
          distance={1}
          position={[0, -0.2, 0]}
        />
      </group>
    </group>
  );
}

function PlaneEye({
  position,
  meshRef,
}: {
  position: [number, number, number];
  meshRef: React.RefObject<THREE.Mesh | null>;
}) {
  return (
    <mesh ref={meshRef} position={position}>
      <planeGeometry args={[0.12, 0.12]} />
      <meshStandardMaterial
        color="#00ffff"
        emissive="#00ffff"
        emissiveIntensity={2}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function CanvasWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Canvas camera={{ position: [0, 1.2, 4.2], fov: 45 }}>{children}</Canvas>
  );
}

function SceneEnvironment() {
  return (
    <>
      <ambientLight intensity={1.2} color="#ffffff" />
      <spotLight
        position={[10, 10, 10]}
        angle={0.15}
        penumbra={1}
        intensity={4}
        color="#ffffff"
        castShadow
      />
      <pointLight position={[-10, -5, -10]} intensity={2} color="#38bdf8" />
      <pointLight position={[0, -5, 5]} intensity={1} color="#ffffff" />
      <DreiOrbitControls
        enablePan={false}
        minPolarAngle={Math.PI / 3}
        maxPolarAngle={Math.PI / 2}
      />
    </>
  );
}

// --- Main Component ---

/**
 * Plays a pre-synthesized narration track (Piper audio + Rhubarb viseme
 * timeline) through the procedural CogniBot avatar. Mouth shape is driven
 * directly off audioRef.current.currentTime against visemeTimeline each
 * frame in RobotModel's useFrame — no React state in that loop, so mouth
 * sync doesn't trigger re-renders.
 */
export default function CourseNarrator({
  audioUrl,
  visemeTimeline,
  animation,
  autoPlay = false,
  isPaused = false,
  onComplete,
}: CourseNarratorProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const resolvedAnimation =
    animation ?? (isPlaying ? "Teacher_Talking" : "Teacher_StandingPose");

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleEnded = () => {
      setIsPlaying(false);
      onComplete?.();
    };
    const handlePlay = () => setIsPlaying(true);
    const handlePauseEvent = () => setIsPlaying(false);

    audio.addEventListener("ended", handleEnded);
    audio.addEventListener("play", handlePlay);
    audio.addEventListener("pause", handlePauseEvent);
    return () => {
      audio.removeEventListener("ended", handleEnded);
      audio.removeEventListener("play", handlePlay);
      audio.removeEventListener("pause", handlePauseEvent);
    };
  }, [onComplete]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !autoPlay) return;
    audio.play().catch((error) => {
      console.warn(
        `[CourseNarrator] autoplay failed for ${audioUrl} (likely needs a user gesture):`,
        error,
      );
    });
    // Re-fires on audioUrl change so advancing to a new slide's narration autoplays too.
  }, [autoPlay, audioUrl]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !isPaused) return;
    audio.pause();
  }, [isPaused]);

  return (
    <div className="w-full h-full relative bg-gradient-to-b from-gray-900 to-black rounded-xl overflow-hidden shadow-2xl border border-cyan-500/30">
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(#00ffff 1px, transparent 1px), linear-gradient(90deg, #00ffff 1px, transparent 1px)",
          backgroundSize: "40px 40px",
        }}
      />

      {/* biome-ignore lint/a11y/useMediaCaption: narration audio for an on-page avatar, not a captioned media asset */}
      <audio ref={audioRef} src={audioUrl} preload="auto" />

      <React.Suspense
        fallback={
          <div className="text-cyan-500 p-10">Initializing Core...</div>
        }
      >
        <CanvasWrapper>
          <RobotModel
            currentAnimation={resolvedAnimation}
            speaking={isPlaying && !isPaused}
            audioRef={audioRef}
            visemeTimeline={visemeTimeline}
          />
          <SceneEnvironment />
        </CanvasWrapper>
      </React.Suspense>

      {/* Status only — no independent controls; play/pause is driven entirely by the isPaused prop */}
      <div className="absolute top-2 left-2 flex items-center space-x-1">
        <div
          className={`w-2 h-2 rounded-full shadow-[0_0_10px_#00ffff] ${isPaused ? "bg-yellow-500 animate-none" : "bg-cyan-500 animate-pulse"}`}
        />
        <span className="text-cyan-500 font-mono text-[9px] tracking-widest">
          {isPaused ? "PAUSED" : "ONLINE"}
        </span>
      </div>
    </div>
  );
}
