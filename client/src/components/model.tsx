'use client';

import React, { useRef, useState, useEffect, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import { RoundedBox, Sphere, Cylinder, Float, Trail, Text, Html } from '@react-three/drei';
import * as THREE from 'three';

// --- Types (Shared with Tutor3D) ---
export type TutorAction =
    | { type: 'animation'; name: string; loop?: boolean; duration?: number; waitForAction?: boolean }
    | { type: 'speak'; text: string; voice?: string }
    | { type: 'wait'; duration: number }
    | { type: 'move'; position: [number, number, number]; duration: number };

export interface TutorScriptProps {
    script: TutorAction[];
    autoPlay?: boolean;
    onComplete?: () => void;
    /** External pause control. When true, script execution halts. */
    isPaused?: boolean;
}

// --- Helper for Speech ---
let currentUtterance: SpeechSynthesisUtterance | null = null;
let voicesLoaded: SpeechSynthesisVoice[] = [];
let voicesReady = false;

const loadVoices = () => {
    if (!window.speechSynthesis) return;
    const voices = window.speechSynthesis.getVoices();
    if (voices.length > 0) {
        voicesLoaded = voices;
        voicesReady = true;
    }
    return voices;
};

// Pre-load voices if available
if (typeof window !== 'undefined') {
    // Chromium needs this hack to kick-start voice loading
    window.speechSynthesis?.getVoices();
    window.speechSynthesis?.addEventListener('voiceschanged', () => {
        loadVoices();
    }, { once: true });
}

const speakText = (text: string, onEnd: () => void) => {
    if (!window.speechSynthesis) {
        console.warn("SpeechSynthesis not available");
        setTimeout(onEnd, 2000);
        return;
    }

    // Chromium bug: speech synthesis can get stuck in a paused state.
    // Calling pause() then resume() "unsticks" it.
    if (window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
    }
    // Also check if it's speaking but stuck
    if (window.speechSynthesis.speaking && !window.speechSynthesis.pending) {
        window.speechSynthesis.cancel();
    }

    // Ensure voices are loaded
    if (!voicesReady) {
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) {
            voicesLoaded = voices;
            voicesReady = true;
        } else {
            // Voices not ready yet, wait a bit and retry
            console.warn("Voices not loaded, retrying in 100ms...");
            setTimeout(() => speakText(text, onEnd), 100);
            return;
        }
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    currentUtterance = utterance;

    // Select a voice - must happen AFTER cancel() in Chromium
    if (voicesLoaded.length > 0) {
        const preferredVoice = voicesLoaded.find(
            v => v.name.includes('Google US English')
        ) || voicesLoaded.find(
            v => v.name.includes('Google') && v.lang.startsWith('en')
        ) || voicesLoaded.find(
            v => v.lang === 'en-US'
        ) || voicesLoaded.find(
            v => v.lang.startsWith('en')
        );
        if (preferredVoice) {
            utterance.voice = preferredVoice;
        }
    }

    utterance.rate = 1.2;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    utterance.onend = () => {
        currentUtterance = null;
        onEnd();
    };

    utterance.onerror = (e) => {
        if (e.error !== 'canceled') {
            console.warn("Speech synthesis error:", e.error, e);
        }
        currentUtterance = null;
        onEnd();
    };

    try {
        window.speechSynthesis.speak(utterance);
    } catch (err) {
        console.error("SpeechSynthesis.speak failed:", err);
        currentUtterance = null;
        setTimeout(onEnd, 1000);
    }
};

// --- Procedural Robot Component ---

interface RobotProps {
    currentAnimation: string;
    speaking: boolean;
}

const RobotModel: React.FC<RobotProps> = ({ currentAnimation, speaking }) => {
    const group = useRef<THREE.Group>(null);
    const headRef = useRef<THREE.Group>(null);
    const bodyRef = useRef<THREE.Mesh>(null);
    const leftArmRef = useRef<THREE.Group>(null);
    const rightArmRef = useRef<THREE.Group>(null);
    const eyesRef = useRef<THREE.Group>(null);
    const mouthRef = useRef<THREE.Mesh>(null);

    // State for smooth transitions
    const targetRotationBody = useRef(0);
    const targetRotationHead = useRef(0);
    const targetArmLeft = useRef([0, 0, 0]); // x, y, z
    const targetArmRight = useRef([0, 0, 0]);
    const eyeColor = useRef(new THREE.Color('#00ffff'));
    const nextBlinkTime = useRef(0);

    // Animation Logic
    useFrame((state, delta) => {
        const time = state.clock.getElapsedTime();
        const t = time * 2;

        // --- 1. Determine Targets based on Animation State ---
        let swaySpeed = 1;
        let swayAmp = 0.1;

        // Reset defaults
        let tHeadRot = 0;
        let tBodyRot = 0;
        let tLeftArm = [0, 0, 0.3]; // Relaxed at side
        let tRightArm = [0, 0, -0.3];
        let tEyeColor = '#00ffff'; // Cyan default
        let tEyeScale = [1, 1, 1]; // Normal eyes

        // Blinking Logic
        let blinkScale = 1;
        if (time > nextBlinkTime.current) {
            const blinkPhase = (time - nextBlinkTime.current) * 10; // Fast blink
            if (blinkPhase < Math.PI) {
                blinkScale = Math.max(0.1, Math.abs(Math.cos(blinkPhase)));
            } else {
                nextBlinkTime.current = time + 2 + Math.random() * 4; // Next blink in 2-6s
            }
        }

        switch (currentAnimation) {
            case 'Teacher_StandingPose':
                // Idle breathing
                tLeftArm = [Math.sin(t) * 0.1, 0, 0.3];
                tRightArm = [Math.sin(t + 1) * 0.1, 0, -0.3];
                break;

            case 'Teacher_ExplainingGestures':
            case 'Teacher_Talking':
                tEyeColor = '#00ffaa'; // Greenish when explaining
                tEyeScale = [1.1, 1.1, 1]; // Slightly wide/engaged

                // Active hands - Dynamic Symmetric Wave
                // Increased amplitude and added complexity
                const wave = Math.sin(t * 4) * 0.35; // Faster and bigger vertical motion
                const waveZ = Math.cos(t * 2.5) * 0.2; // Side-to-side variation

                // Arms reach out more (more negative X) and move more
                tLeftArm = [-0.8 + wave, 0.5, 0.6 + waveZ];
                tRightArm = [-0.8 + wave, -0.5, -0.6 - waveZ];

                // Increased body sway for energy
                tBodyRot = Math.sin(t * 1.5) * 0.15;
                tHeadRot = Math.sin(t * 2) * 0.05;
                break;

            case 'Teacher_ThinkingPose':
                tEyeColor = '#ffaa00'; // Orange thinking
                tEyeScale = [1, 0.3, 1]; // Squint/Thinking
                tHeadRot = 0.3; // Tilt head
                // Hand to chin: High Forward (Neg X), Inwards (Pos Z for Right Arm)
                tRightArm = [-2.2, -0.5, 0.8];
                tLeftArm = [0, 0, 0.3];
                break;

            case 'Teacher_PointingBoard':
                tEyeColor = '#ff00ff'; // Magenta pointing
                tBodyRot = -0.2; // Turn slightly
                // Point Out: Forward (Neg X), Outwards (Neg Z for Right Arm)
                tRightArm = [-1.3, 0, -0.8];
                tHeadRot = -0.4; // Look at point
                break;

            case 'Teacher_PointScreen':
                // Model is on the RIGHT, Screen is on the LEFT (Viewer's Left)
                // So Model points with LEFT arm to its RIGHT (Viewer's Left)
                tEyeColor = '#ff00ff';
                tBodyRot = 0.3; // Turn towards screen
                tHeadRot = 0.4; // Look at screen
                tLeftArm = [-1.3, 0, 0.8]; // Point Left Arm towards screen
                tRightArm = [0, 0, -0.3]; // Relax Right
                break;

            case 'Teacher_SwipeNext':
                tEyeColor = '#00ffff';
                tEyeScale = [1.2, 1, 1]; // Focused
                // Swipe gesture with Right Arm
                tRightArm = [-0.5, -0.5 + Math.sin(t * 5) * 0.5, -0.5 + Math.cos(t * 5) * 0.5];
                tHeadRot = -0.1;
                break;

            case 'Teacher_Emphasize':
                tEyeColor = '#ffaa00'; // Orange intensity
                tEyeScale = [1.2, 1.2, 1]; // Wide eyes

                // Stronger chopping motion
                const chop = Math.sin(t * 8) * 0.35; // Increased amplitude

                // Arms raised higher (-1.0) and chopping vigorously
                tLeftArm = [-1.0 + chop, 0.5, 0.4];
                tRightArm = [-1.0 + chop, -0.5, -0.4];

                // Head nods in sync with emphasis
                tHeadRot = Math.sin(t * 8) * 0.1;
                tBodyRot = 0;
                break;

            case 'Teacher_Listening':
                tEyeColor = '#00ffff';
                tEyeScale = [1, 0.8, 1]; // Relaxed
                // Hands clasped/relaxed in front
                tLeftArm = [-0.5, -0.5, 0.5];
                tRightArm = [-0.5, 0.5, -0.5];
                tHeadRot = Math.sin(t * 0.5) * 0.05;
                break;

            case 'Teacher_EncouragingNod':
                tEyeColor = '#00ff00'; // Green happy
                tEyeScale = [1.1, 0.6, 1]; // Happy squint
                tHeadRot = Math.sin(t * 10) * 0.1; // Nodding
                tLeftArm = [0.5, 0, 0.5]; // Open arms
                tRightArm = [0.5, 0, -0.5];
                break;

            case 'Teacher_LookingAround':
                tHeadRot = Math.sin(t) * 0.5;
                break;
        }

        // Override for speaking (mouth/head bob)
        let tMouthScaleY = 0.05; // Default thin line
        if (speaking) {
            // Rapid head bob or intensity change
            const speakBob = Math.sin(time * 15) * 0.05;
            tHeadRot += speakBob;

            // Mouth animation
            tMouthScaleY = 0.15 + Math.sin(time * 20) * 0.1;
        }

        // --- 2. Lerp to Targets ---
        const lerpFactor = 5 * delta;

        // Body Rotation
        if (group.current) {
            group.current.rotation.y = THREE.MathUtils.lerp(group.current.rotation.y, tBodyRot, lerpFactor);
        }

        // Head
        if (headRef.current) {
            headRef.current.rotation.y = THREE.MathUtils.lerp(headRef.current.rotation.y, tHeadRot, lerpFactor);
            headRef.current.rotation.z = THREE.MathUtils.lerp(headRef.current.rotation.z, Math.sin(t * 0.5) * 0.05, lerpFactor); // Idle tilt
        }

        // Arms
        if (leftArmRef.current) {
            leftArmRef.current.rotation.x = THREE.MathUtils.lerp(leftArmRef.current.rotation.x, tLeftArm[0], lerpFactor);
            leftArmRef.current.rotation.y = THREE.MathUtils.lerp(leftArmRef.current.rotation.y, tLeftArm[1], lerpFactor);
            leftArmRef.current.rotation.z = THREE.MathUtils.lerp(leftArmRef.current.rotation.z, tLeftArm[2], lerpFactor);
        }
        if (rightArmRef.current) {
            rightArmRef.current.rotation.x = THREE.MathUtils.lerp(rightArmRef.current.rotation.x, tRightArm[0], lerpFactor);
            rightArmRef.current.rotation.y = THREE.MathUtils.lerp(rightArmRef.current.rotation.y, tRightArm[1], lerpFactor);
            rightArmRef.current.rotation.z = THREE.MathUtils.lerp(rightArmRef.current.rotation.z, tRightArm[2], lerpFactor);
        }

        // Eyes Color & Scale
        eyeColor.current.lerp(new THREE.Color(tEyeColor), lerpFactor);
        if (eyesRef.current) {
            // Apply scale (Expression * Blink)
            const finalScaleY = tEyeScale[1] * blinkScale;
            eyesRef.current.scale.set(
                THREE.MathUtils.lerp(eyesRef.current.scale.x, tEyeScale[0], lerpFactor),
                THREE.MathUtils.lerp(eyesRef.current.scale.y, finalScaleY, lerpFactor * 2), // Faster blink
                1
            );

            (eyesRef.current.children[0] as any).material.color = eyeColor.current;
            (eyesRef.current.children[0] as any).material.emissive = eyeColor.current;
            (eyesRef.current.children[1] as any).material.color = eyeColor.current;
            (eyesRef.current.children[1] as any).material.emissive = eyeColor.current;
        }

        // Mouth Scale
        if (mouthRef.current) {
            mouthRef.current.scale.y = THREE.MathUtils.lerp(mouthRef.current.scale.y, tMouthScaleY, lerpFactor * 4);
            // Change mouth color based on speaking?
            (mouthRef.current.material as any).color = eyeColor.current;
            (mouthRef.current.material as any).emissive = eyeColor.current;
        }

        // Floating motion
        if (group.current) {
            group.current.position.y = -0.6 + Math.sin(time) * 0.1;
        }
    });

    return (
        <group ref={group}>
            {/* --- HEAD --- */}
            <group ref={headRef} position={[0, 1.4, 0]}>
                {/* Main Head Shape */}
                <RoundedBox args={[0.5, 0.4, 0.45]} radius={0.1} smoothness={4}>
                    <meshStandardMaterial color="#ffffff" emissive="#222222" metalness={0.6} roughness={0.2} />
                </RoundedBox>

                {/* Face Screen (Black Glass) */}
                <RoundedBox args={[0.42, 0.3, 0.05]} radius={0.05} position={[0, 0, 0.21]}>
                    <meshStandardMaterial color="#222222" metalness={0.9} roughness={0.1} />
                </RoundedBox>

                {/* Eyes */}
                <group ref={eyesRef} position={[0, 0, 0.24]}>
                    {/* Left Eye */}
                    <PlaneEye position={[-0.1, 0, 0]} />
                    {/* Right Eye */}
                    <PlaneEye position={[0.1, 0, 0]} />
                </group>

                {/* Mouth */}
                <mesh ref={mouthRef} position={[0, -0.12, 0.24]}>
                    <planeGeometry args={[0.15, 0.05]} />
                    <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={1} />
                </mesh>

                {/* Antenna / Halo */}
                <group position={[0, 0.3, 0]}>
                    <Cylinder args={[0.02, 0.02, 0.2]} position={[0, 0, 0]}>
                        <meshStandardMaterial color="#888" />
                    </Cylinder>
                    <Sphere args={[0.05]} position={[0, 0.1, 0]}>
                        <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={2} />
                    </Sphere>
                </group>
            </group>

            {/* --- NECK --- */}
            <Cylinder args={[0.05, 0.05, 0.2]} position={[0, 1.15, 0]}>
                <meshStandardMaterial color="#0ea5e9" emissive="#004488" emissiveIntensity={0.5} />
            </Cylinder>

            {/* --- BODY --- */}
            <group ref={bodyRef} position={[0, 0.6, 0]}>
                {/* Upper Body */}
                <RoundedBox args={[0.6, 0.7, 0.4]} radius={0.2} smoothness={4} position={[0, 0.1, 0]}>
                    <meshStandardMaterial color="#ffffff" emissive="#222222" metalness={0.5} roughness={0.2} />
                </RoundedBox>
                {/* Core Reactor */}
                <Cylinder args={[0.15, 0.15, 0.05]} rotation={[Math.PI / 2, 0, 0]} position={[0, 0.1, 0.2]}>
                    <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={1} />
                </Cylinder>
            </group>

            {/* --- ARMS --- */}
            {/* Left Arm Pivot */}
            <group position={[0.35, 0.9, 0]}>
                <group ref={leftArmRef}>
                    {/* Shoulder */}
                    <Sphere args={[0.12]} position={[0, 0, 0]}>
                        <meshStandardMaterial color="#0ea5e9" metalness={0.8} roughness={0.2} />
                    </Sphere>
                    {/* Arm Segment */}
                    <RoundedBox args={[0.1, 0.4, 0.1]} position={[0, -0.25, 0]}>
                        <meshStandardMaterial color="#ffffff" emissive="#111111" />
                    </RoundedBox>
                    {/* Hand */}
                    <Sphere args={[0.1]} position={[0, -0.5, 0]}>
                        <meshStandardMaterial color="#0ea5e9" metalness={0.8} roughness={0.2} />
                    </Sphere>
                </group>
            </group>

            {/* Right Arm Pivot */}
            <group position={[-0.35, 0.9, 0]}>
                <group ref={rightArmRef}>
                    <Sphere args={[0.12]} position={[0, 0, 0]}>
                        <meshStandardMaterial color="#0ea5e9" metalness={0.8} roughness={0.2} />
                    </Sphere>
                    <RoundedBox args={[0.1, 0.4, 0.1]} position={[0, -0.25, 0]}>
                        <meshStandardMaterial color="#ffffff" emissive="#111111" />
                    </RoundedBox>
                    <Sphere args={[0.1]} position={[0, -0.5, 0]}>
                        <meshStandardMaterial color="#0ea5e9" metalness={0.8} roughness={0.2} />
                    </Sphere>
                </group>
            </group>

            {/* --- HOVER ENGINE / BASE --- */}
            <group position={[0, 0.1, 0]}>
                <Cylinder args={[0.1, 0.2, 0.3]} position={[0, 0, 0]}>
                    <meshStandardMaterial color="#334155" />
                </Cylinder>
                {/* Thruster Glow */}
                <pointLight color="#00ffff" intensity={1} distance={1} position={[0, -0.2, 0]} />
            </group>
        </group>
    );
};

const PlaneEye = ({ position }: { position: [number, number, number] }) => (
    <mesh position={position}>
        <planeGeometry args={[0.12, 0.12]} />
        <meshStandardMaterial color="#00ffff" emissive="#00ffff" emissiveIntensity={2} side={THREE.DoubleSide} />
    </mesh>
);


// --- Main Component ---

const CogniBot: React.FC<TutorScriptProps> = ({
    script,
    autoPlay = false,
    onComplete
}) => {
    const [currentStep, setCurrentStep] = useState(-1);
    const [currentAnimation, setCurrentAnimation] = useState('Teacher_StandingPose');
    const [speechText, setSpeechText] = useState<string | null>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const timerRef = useRef<NodeJS.Timeout | null>(null);

    const processStep = useRef<(index: number) => void>(null);

    // Define processStep in a way that it can be called recursively
    useEffect(() => {
        processStep.current = (index: number) => {
            if (isPaused) return; // Stop if paused

            if (index >= script.length) {
                setIsPlaying(false);
                onComplete?.();
                return;
            }

            const action = script[index];
            setCurrentStep(index);

            // Clear any existing timer
            if (timerRef.current) clearTimeout(timerRef.current);

            switch (action.type) {
                case 'animation':
                    setCurrentAnimation(action.name);
                    // If waitForAction is explicitly false, move to next step immediately
                    if (action.waitForAction === false) {
                        processStep.current?.(index + 1);
                    } else {
                        // Otherwise wait for duration or default
                        const waitTime = action.duration || 1500;
                        timerRef.current = setTimeout(() => {
                            if (!isPaused) processStep.current?.(index + 1);
                        }, waitTime);
                    }
                    break;

                case 'speak':
                    setSpeechText(action.text);
                    speakText(action.text, () => {
                        setSpeechText(null);
                        if (!isPaused) processStep.current?.(index + 1);
                    });
                    break;

                case 'wait':
                    timerRef.current = setTimeout(() => {
                        if (!isPaused) processStep.current?.(index + 1);
                    }, action.duration);
                    break;

                default:
                    processStep.current?.(index + 1);
            }
        };
    }, [script, onComplete, isPaused]); // Re-create when paused state changes

    // Effect to resume when unpaused
    useEffect(() => {
        if (!isPaused && isPlaying && processStep.current) {
            const currentAction = script[currentStep];
            // Only restart step if it's NOT speech (speech resumes via window.speechSynthesis.resume)
            // OR if speech was somehow cancelled/finished.
            // Actually, simplest is: if it was a timer-based step, we must restart it.
            // If it was speech, we just resume audio.
            if (currentAction && currentAction.type !== 'speak') {
                processStep.current(currentStep);
            } else {
                window.speechSynthesis.resume();
            }
        }
    }, [isPaused]);

    useEffect(() => {
        if (autoPlay && script.length > 0 && !isPlaying && processStep.current && !isPaused) {
            setIsPlaying(true);
            processStep.current(0);
        }
    }, [autoPlay, script, isPlaying, isPaused]);

    // Cleanup speech on unmount or script change
    useEffect(() => {
        return () => {
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
            if (timerRef.current) {
                clearTimeout(timerRef.current);
            }
        };
    }, [script]);

    const togglePause = () => {
        if (isPaused) {
            setIsPaused(false);
            // Resume handled by effect
        } else {
            window.speechSynthesis.pause();
            if (timerRef.current) clearTimeout(timerRef.current);
            setIsPaused(true);
        }
    };

    return (
        <div className="w-full h-full relative min-h-[400px] bg-gradient-to-b from-gray-900 to-black rounded-xl overflow-hidden shadow-2xl border border-cyan-500/30">
            {/* Tech Grid Background Overlay */}
            <div className="absolute inset-0 opacity-20 pointer-events-none"
                style={{ backgroundImage: 'linear-gradient(#00ffff 1px, transparent 1px), linear-gradient(90deg, #00ffff 1px, transparent 1px)', backgroundSize: '40px 40px' }}>
            </div>

            {/* Canvas */}
            {/* Added OrbitControls import to the top if not present, but it is */}
            {/* Need to make sure we import OrbitControls from drei */}
            <React.Suspense fallback={<div className="text-cyan-500 p-10">Initializing Core...</div>}>
                {/* We need a Canvas here. The RobotModel is just the scene content. */}
                {/* Note: In the previous file I put Canvas inside the component. I should do the same here. */}
                <CanvasWrapper>
                    <RobotModel currentAnimation={currentAnimation} speaking={!!speechText} />
                    <Environment />
                </CanvasWrapper>
            </React.Suspense>

            {/* Overlay UI */}
            <div className="absolute top-4 left-4 flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full shadow-[0_0_10px_#00ffff] ${isPaused ? 'bg-yellow-500 animate-none' : 'bg-cyan-500 animate-pulse'}`}></div>
                    <span className="text-cyan-500 font-mono text-xs tracking-widest">{isPaused ? 'SYSTEM PAUSED' : 'COGNIBOT ONLINE'}</span>
                </div>

                <button
                    onClick={togglePause}
                    className="bg-cyan-950/80 hover:bg-cyan-900 text-cyan-400 border border-cyan-500/50 px-3 py-1 rounded text-xs font-mono transition-all uppercase"
                >
                    {isPaused ? '▶ Resume' : '⏸ Pause'}
                </button>
            </div>

            <div className="absolute bottom-4 right-4 pointer-events-none">
                <div className="bg-black/60 backdrop-blur-md p-2 rounded-lg border border-cyan-500/30 text-cyan-400/60 text-[10px] font-mono text-right">
                    {isPlaying ? `SEQ: ${currentStep + 1}/${script.length}` : 'READY'}
                    {currentAnimation && <div>ACT: {currentAnimation}</div>}
                </div>
            </div>
        </div>
    );
};

// Helper wrapper to provide Canvas context
import { Canvas } from '@react-three/fiber';
import { OrbitControls as DreiOrbitControls } from '@react-three/drei';

const CanvasWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <Canvas camera={{ position: [0, 1.2, 4.2], fov: 45 }}>
        {children}
    </Canvas>
);

export default CogniBot;

const Environment = () => (
    <>
        <ambientLight intensity={1.2} color="#ffffff" />
        <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={4} color="#ffffff" castShadow />
        <pointLight position={[-10, -5, -10]} intensity={2} color="#38bdf8" />
        <pointLight position={[0, -5, 5]} intensity={1} color="#ffffff" />
        <DreiOrbitControls enablePan={false} minPolarAngle={Math.PI / 3} maxPolarAngle={Math.PI / 2} />
        <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5} />
    </>
);
