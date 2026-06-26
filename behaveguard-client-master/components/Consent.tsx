"use client";

import { useState } from "react";

export default function Consent({ onAgree }: { onAgree: () => void }) {
  const [checked, setChecked] = useState(false);

  return (
    <div className="flex-1 flex items-center justify-center px-6">
      <div className="max-w-lg w-full fade-up">
        <div className="font-mono-tight text-xs uppercase tracking-[0.3em] text-cyan mb-3">
          before we start
        </div>
        <h2 className="text-2xl font-semibold mb-5">What this test records</h2>

        <div className="space-y-3 mb-6">
          <Row label="Keystroke timing" detail="When keys are pressed and released — not which characters you type." />
          <Row label="Mouse movement" detail="Cursor position, speed, and click accuracy during two short tasks." />
          <Row label="No personal data" detail="Only the name you enter next, used as a session label." />
        </div>

        <div className="bg-surface border border-border rounded-lg p-4 text-sm text-muted leading-relaxed mb-6">
          This data is used to build behavioral-biometric models. It is stored in a
          private research spreadsheet, not shared publicly, and not linked to any
          account. You can close this tab at any time before the final screen to opt out —
          nothing is sent until the test is complete.
        </div>

        <label className="flex items-start gap-3 mb-6 cursor-pointer group">
          <input
            type="checkbox"
            checked={checked}
            onChange={(e) => setChecked(e.target.checked)}
            className="mt-1 w-4 h-4 accent-amber"
          />
          <span className="text-sm text-text group-hover:text-amber transition">
            I understand what is being recorded and agree to take part.
          </span>
        </label>

        <button
          disabled={!checked}
          onClick={onAgree}
          className="w-full font-mono-tight text-sm uppercase tracking-wider bg-amber text-bg px-8 py-3 rounded-md disabled:opacity-30 disabled:cursor-not-allowed hover:brightness-110 transition active:scale-[0.98]"
        >
          agree & continue →
        </button>
      </div>
    </div>
  );
}

function Row({ label, detail }: { label: string; detail: string }) {
  return (
    <div className="flex gap-3 items-start bg-surface/60 border border-border rounded-lg p-3">
      <div className="w-1.5 h-1.5 rounded-full bg-cyan mt-2 shrink-0" />
      <div>
        <div className="text-sm font-medium text-text">{label}</div>
        <div className="text-xs text-muted mt-0.5">{detail}</div>
      </div>
    </div>
  );
}
