"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { DragTrial } from "@/lib/types";

const N_DRAGS = 10;
const CHIP_SIZE = 44;
const ZONE_SIZE = 90;

function randomNonOverlapping(w: number, h: number) {
  const start = { x: 60, y: h / 2 };
  const zone = {
    x: w - ZONE_SIZE - 40 + Math.random() * 20,
    y: 60 + Math.random() * (h - 180),
  };
  return { start, zone };
}

export default function MouseDragTask({ onComplete }: { onComplete: (trials: DragTrial[]) => void }) {
  const areaRef = useRef<HTMLDivElement>(null);
  const [trialIdx, setTrialIdx] = useState(0);
  const [chip, setChip] = useState({ x: 60, y: 200 });
  const [zone, setZone] = useState({ x: 300, y: 200 });
  const [dragging, setDragging] = useState(false);
  const [success, setSuccess] = useState<boolean | null>(null);
  const trialsRef = useRef<DragTrial[]>([]);
  const startedAt = useRef(0);
  const startPos = useRef({ x: 0, y: 0 });
  const pathRef = useRef<{ x: number; y: number; ts: number }[]>([]);
  const lastSample = useRef(0);

  const layout = useCallback(() => {
    const area = areaRef.current;
    if (!area) return;
    const { width, height } = area.getBoundingClientRect();
    const { start, zone } = randomNonOverlapping(width, height);
    setChip(start);
    setZone(zone);
  }, []);

  useEffect(() => {
    layout();
  }, [layout, trialIdx]);

  function handlePointerDown(e: React.PointerEvent) {
    e.preventDefault();
    setDragging(true);
    startedAt.current = performance.now();
    startPos.current = { ...chip };
    pathRef.current = [];
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }

  function handlePointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragging) return;
    const rect = areaRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    setChip({ x, y });
    const now = performance.now();
    if (now - lastSample.current >= 16) {
      lastSample.current = now;
      pathRef.current.push({ x, y, ts: now });
    }
  }

  function handlePointerUp() {
    if (!dragging) return;
    setDragging(false);
    const endedAt = performance.now();
    const dist = Math.hypot(chip.x - (zone.x + ZONE_SIZE / 2), chip.y - (zone.y + ZONE_SIZE / 2));
    const ok = dist < ZONE_SIZE / 2;
    const trial: DragTrial = {
      start_x: startPos.current.x,
      start_y: startPos.current.y,
      end_x: chip.x,
      end_y: chip.y,
      zone_x: zone.x,
      zone_y: zone.y,
      started_at: startedAt.current,
      ended_at: endedAt,
      duration_ms: endedAt - startedAt.current,
      success: ok,
      path: pathRef.current,
    };
    trialsRef.current.push(trial);
    setSuccess(ok);
    setTimeout(() => {
      setSuccess(null);
      if (trialIdx + 1 >= N_DRAGS) {
        onComplete(trialsRef.current);
      } else {
        setTrialIdx((i) => i + 1);
      }
    }, ok ? 250 : 450);
    if (!ok) setChip(startPos.current);
  }

  const progressPct = (trialIdx / N_DRAGS) * 100;

  return (
    <div className="flex-1 flex flex-col px-6 pb-6">
      <div className="max-w-3xl w-full mx-auto fade-up flex-1 flex flex-col">
        <div className="flex items-center justify-between mb-3 pt-2">
          <span className="font-mono-tight text-xs uppercase tracking-[0.3em] text-cyan">
            task 2 · drag into the zone
          </span>
          <span className="font-mono-tight text-sm text-muted tabular-nums">
            {trialIdx}/{N_DRAGS}
          </span>
        </div>
        <div className="h-1 w-full bg-border rounded-full mb-4 overflow-hidden">
          <div className="h-full bg-cyan transition-all duration-200" style={{ width: `${progressPct}%` }} />
        </div>

        <div
          ref={areaRef}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          className="relative flex-1 min-h-[420px] bg-surface border border-border rounded-xl overflow-hidden touch-none"
        >
          <p className="absolute top-3 left-1/2 -translate-x-1/2 text-xs text-muted font-mono-tight">
            drag the chip into the ring
          </p>
          <div
            style={{ left: zone.x, top: zone.y, width: ZONE_SIZE, height: ZONE_SIZE }}
            className={`absolute rounded-full border-2 transition-colors ${
              success === true ? "border-cyan bg-cyan/10" : success === false ? "border-danger" : "border-border"
            }`}
          />
          <div
            onPointerDown={handlePointerDown}
            style={{
              left: chip.x - CHIP_SIZE / 2,
              top: chip.y - CHIP_SIZE / 2,
              width: CHIP_SIZE,
              height: CHIP_SIZE,
            }}
            className={`absolute rounded-lg cursor-grab active:cursor-grabbing flex items-center justify-center transition-colors ${
              dragging ? "bg-amber" : "bg-cyan"
            }`}
          >
            <div className="w-3 h-3 rounded-sm bg-bg/40" />
          </div>
        </div>
      </div>
    </div>
  );
}
