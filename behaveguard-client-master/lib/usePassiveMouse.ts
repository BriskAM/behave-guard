"use client";

import { useEffect, useRef } from "react";
import { MousePoint } from "./types";

export function usePassiveMouseCollector(isEnabled: boolean = true) {
  const pointsRef = useRef<MousePoint[]>([]);
  const last = useRef<{ x: number; y: number; ts: number } | null>(null);

  useEffect(() => {
    if (!isEnabled) {
      last.current = null;
      return;
    }

    function onMove(e: MouseEvent) {
      const ts = performance.now();
      const x = e.clientX;
      const y = e.clientY;
      const prev = last.current;
      if (prev) {
        const dx = x - prev.x;
        const dy = y - prev.y;
        const dt = ts - prev.ts;
        if (Math.hypot(dx, dy) < 5 && dt < 16) return;
        pointsRef.current.push({ x, y, ts, dx, dy });
      } else {
        pointsRef.current.push({ x, y, ts, dx: 0, dy: 0 });
      }
      last.current = { x, y, ts };
      // cap to ~3000 points to bound memory/payload, matching the spec's target density
      if (pointsRef.current.length > 3000) pointsRef.current.shift();
    }
    window.addEventListener("mousemove", onMove, { passive: true });
    return () => window.removeEventListener("mousemove", onMove);
  }, [isEnabled]);

  return pointsRef;
}
