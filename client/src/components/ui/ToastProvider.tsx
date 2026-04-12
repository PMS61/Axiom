"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

type ToastType = "info" | "success" | "error";

interface ToastItem {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  addToast: (message: string, type?: ToastType, durationMs?: number) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const addToast = useCallback(
    (message: string, type: ToastType = "info", durationMs = 4200) => {
      const id = `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      setToasts((prev) => [...prev, { id, message, type }]);

      window.setTimeout(() => {
        removeToast(id);
      }, durationMs);
    },
    [removeToast],
  );

  const value = useMemo(() => ({ addToast }), [addToast]);

  return (
    <ToastContext.Provider value={value}>
      {children}

      <div
        aria-live="polite"
        aria-atomic="true"
        style={{
          position: "fixed",
          top: 76,
          right: 16,
          zIndex: 100001,
          display: "flex",
          flexDirection: "column",
          gap: 10,
          pointerEvents: "none",
          width: "min(420px, calc(100vw - 32px))",
        }}
      >
        {toasts.map((toast) => {
          const borderColor =
            toast.type === "success"
              ? "var(--safe)"
              : toast.type === "error"
                ? "var(--vermillion)"
                : "var(--ink)";

          const label =
            toast.type === "success"
              ? "SUCCESS"
              : toast.type === "error"
                ? "ERROR"
                : "INFO";

          return (
            <div
              key={toast.id}
              style={{
                pointerEvents: "auto",
                border: `0.5px solid ${borderColor}`,
                background: "var(--card-bg)",
                color: "var(--ink)",
                boxShadow: "0 10px 24px rgba(0, 0, 0, 0.08)",
                padding: "10px 12px",
              }}
            >
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 9,
                  letterSpacing: "0.12em",
                  color: borderColor,
                  marginBottom: 4,
                }}
              >
                {label}
              </div>
              <div
                style={{
                  fontFamily: "var(--mono)",
                  fontSize: 12,
                  lineHeight: 1.5,
                }}
              >
                {toast.message}
              </div>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    throw new Error("useToast must be used within ToastProvider");
  }
  return ctx;
}
