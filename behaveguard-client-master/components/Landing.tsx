"use client";

export default function Landing({ onStart, onVerify }: { onStart: () => void; onVerify: () => void }) {
  return (
    <div className="flex-1 flex items-center justify-center px-6">
      <div className="max-w-xl text-center fade-up">
        <div className="font-mono-tight text-sm uppercase tracking-[0.3em] text-muted mb-6">
          behaveguard
        </div>
        <h1 className="font-mono-tight text-4xl sm:text-5xl leading-tight mb-4">
          <span className="text-amber">type</span>
          <span className="text-muted">.</span>{" "}
          <span className="text-cyan">move</span>
          <span className="text-muted">.</span>{" "}
          <span className="text-text">be measured</span>
          <span className="caret text-text">|</span>
        </h1>
        <p className="text-muted leading-relaxed mb-10">
          A 5-minute test of how you type and how you move a cursor. Used to study
          behavioral biometrics — the rhythm that's unique to you, not what you type.
        </p>
        <div className="flex flex-col sm:flex-row justify-center items-center gap-4">
          <button
            onClick={onStart}
            className="w-full sm:w-auto font-mono-tight text-sm uppercase tracking-wider bg-amber text-bg px-8 py-3 rounded-md hover:brightness-110 transition active:scale-[0.98]"
          >
            start enrollment test →
          </button>
          <button
            onClick={onVerify}
            className="w-full sm:w-auto font-mono-tight text-sm uppercase tracking-wider border border-cyan text-cyan px-8 py-3 rounded-md hover:bg-cyan/10 transition active:scale-[0.98]"
          >
            verify & identify →
          </button>
        </div>
        <div className="mt-6 text-xs text-muted font-mono-tight">
          ~3 min keyboard · ~2 min mouse
        </div>
      </div>
    </div>
  );
}
