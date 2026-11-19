"use client";

import { useEffect, useState } from "react";

/** Returns false on the first paint and true one frame later, so CSS transitions run. */
export function useDrawn(): boolean {
  const [drawn, setDrawn] = useState(false);
  useEffect(() => {
    const id = window.requestAnimationFrame(() => setDrawn(true));
    return () => window.cancelAnimationFrame(id);
  }, []);
  return drawn;
}
