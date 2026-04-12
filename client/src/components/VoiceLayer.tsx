"use client";

import { useMicVAD } from "@ricky0123/vad-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { parseVoiceCommand } from "@/app/actions/voice";
import { useApp } from "@/lib/store";
import type { EnergyLevel } from "@/lib/types";

const WAKE_WORD = "axiom";
const WAKE_WORD_REGEX = /\b(?:hey|ok|okay)?\s*axiom\b/i;

function clampEnergyLevel(value: number): EnergyLevel {
  if (value <= -2) return -2;
  if (value >= 2) return 2;
  if (value >= 1) return 1;
  if (value <= -1) return -1;
  return 0;
}

function extractWakeCommand(transcript: string): string | null {
  const match = WAKE_WORD_REGEX.exec(transcript);
  if (!match) return null;

  const command = transcript
    .slice(match.index + match[0].length)
    .replace(/^[\s,.:;-]+/, "")
    .trim();

  return command;
}

export default function VoiceLayer() {
  const { state, dispatch } = useApp();
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const processingRef = useRef(false);
  const [vadActive, setVadActive] = useState(false);
  const [agentReply, setAgentReply] = useState<string>("");

  // TTS helper
  const speak = useCallback((text: string) => {
    setAgentReply(text);
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  }, []);

  const handleFinalTranscript = useCallback(
    async (transcript: string) => {
      if (!transcript.trim() || processingRef.current) return;

      const command = extractWakeCommand(transcript);
      if (command === null) {
        setAgentReply(`Say "${WAKE_WORD}" to activate voice commands.`);
        dispatch({ type: "SET_VOICE_TRANSCRIPT", payload: "" });
        return;
      }
      if (!command) {
        setAgentReply("Wake word detected. What should I do?");
        dispatch({ type: "SET_VOICE_TRANSCRIPT", payload: "" });
        return;
      }

      processingRef.current = true;

      try {
        const result = await parseVoiceCommand(command, state.tasks);

        switch (result.type) {
          case "SET_ENERGY": {
            const nextEnergy = clampEnergyLevel(result.payload);
            dispatch({ type: "SET_ENERGY", payload: nextEnergy });
            dispatch({
              type: "ADAPTIVE_RESCHEDULE",
              payload: { reason: "energy_changed" },
            });
            speak(`Setting energy to ${nextEnergy}. Rescheduling.`);
            break;
          }
          case "UPDATE_TASK_STATE":
            dispatch({
              type: "UPDATE_TASK_STATE",
              payload: result.payload,
            });
            speak(`Task updated.`);
            break;
          case "ADAPTIVE_RESCHEDULE":
            dispatch({
              type: "ADAPTIVE_RESCHEDULE",
              payload: { reason: "energy_changed" },
            });
            speak(`Okay, I'm recalculating your schedule.`);
            break;
          case "ASK_FOCUS": {
            const nextTask = state.tasks.find((t) => t.state === "scheduled");
            if (nextTask) {
              speak(`You should focus on ${nextTask.name} next.`);
            } else {
              speak(`You have no tasks scheduled right now.`);
            }
            break;
          }
          default:
            speak(`I heard: ${command}. But I don't know how to handle that.`);
        }
      } catch (err) {
        console.error("Error handling transcript:", err);
      } finally {
        processingRef.current = false;
        dispatch({ type: "SET_VOICE_TRANSCRIPT", payload: "" });
      }
    },
    [state.tasks, dispatch, speak],
  );

  // VAD for activity detection
  useMicVAD({
    startOnLoad: state.isVoiceActive,
    onSpeechStart: () => {
      setVadActive(true);
    },
    onSpeechEnd: () => {
      setVadActive(false);
    },
    baseAssetPath: "/",
    onnxWASMBasePath: "/",
  });

  // Initialize Speech Recognition
  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognitionClass =
        globalThis.SpeechRecognition || globalThis.webkitSpeechRecognition;
      if (SpeechRecognitionClass) {
        const recognition = new SpeechRecognitionClass();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = "en-US";

        recognition.onresult = (event: SpeechRecognitionEvent) => {
          let interimTranscript = "";
          for (let i = event.resultIndex; i < event.results.length; ++i) {
            const result = event.results[i];
            if (result.isFinal) {
              handleFinalTranscript(result[0].transcript);
            } else {
              interimTranscript += result[0].transcript;
            }
          }
          if (interimTranscript) {
            dispatch({
              type: "SET_VOICE_TRANSCRIPT",
              payload: interimTranscript,
            });
          }
        };

        recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
          console.error("Speech recognition error", event.error);
          if (event.error === "no-speech") return;
        };

        recognition.onend = () => {
          if (state.isVoiceActive) {
            try {
              recognition.start();
            } catch {
              // ignore already started
            }
          }
        };

        recognitionRef.current = recognition;
      }
    }
  }, [state.isVoiceActive, handleFinalTranscript, dispatch]);

  useEffect(() => {
    if (state.isVoiceActive) {
      try {
        recognitionRef.current?.start();
        speak(`Voice agent active. Say ${WAKE_WORD} followed by your command.`);
      } catch {
        // ignore
      }
    } else {
      recognitionRef.current?.stop();
    }
  }, [state.isVoiceActive, speak]);

  if (!state.isVoiceActive) return null;

  return (
    <div
      className="voice-overlay"
      style={{
        position: "fixed",
        bottom: 24,
        right: 24,
        background: "var(--bg)",
        color: "var(--ink)",
        padding: "16px",
        borderRadius: 12,
        width: 320,
        zIndex: 1000,
        boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
        border: "1px solid var(--rule)",
        fontFamily: "var(--font-mono)",
        fontSize: 12,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      {/* Header Row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div
            className="pulse-mic"
            style={{
              width: 10,
              height: 10,
              background: processingRef.current
                ? "var(--vermillion)"
                : vadActive
                  ? "#3FB950"
                  : "#8B949E",
              borderRadius: "50%",
              animation: vadActive ? "pulse 1s infinite" : "none",
              transition: "background 0.2s",
            }}
          />
          <div
            style={{
              color: "var(--ink)",
              fontSize: 12,
              fontWeight: "bold",
              textTransform: "uppercase",
              fontFamily: "var(--font-mono)",
            }}
          >
            Wake Word: AXIOM
          </div>
        </div>
        <button
          type="button"
          onClick={() => dispatch({ type: "SET_VOICE_ACTIVE", payload: false })}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "var(--ink)",
            opacity: 0.6,
          }}
        >
          ✕
        </button>
      </div>

      {/* User Transcript */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ fontSize: 10, opacity: 0.5, textTransform: "uppercase" }}>
          You
        </div>
        <div
          style={{
            fontStyle: "italic",
            opacity: state.voiceTranscript ? 1 : 0.6,
            minHeight: 18,
          }}
        >
          {state.voiceTranscript || "Listening for wake word..."}
        </div>
      </div>

      {/* Agent Processing / Reply */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div
          style={{
            fontSize: 10,
            opacity: 0.5,
            textTransform: "uppercase",
            color: "var(--vermillion)",
          }}
        >
          Axiom
        </div>
        <div style={{ minHeight: 18 }}>
          {processingRef.current ? (
            <span style={{ opacity: 0.6 }}>Thinking...</span>
          ) : agentReply ? (
            <span>{agentReply}</span>
          ) : (
            <span style={{ opacity: 0.4 }}>Awaiting input</span>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes pulse {
          0% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.4); opacity: 0.5; }
          100% { transform: scale(1); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
