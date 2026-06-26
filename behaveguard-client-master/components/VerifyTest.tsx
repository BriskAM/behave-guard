"use client";

import { useEffect, useRef, useState } from "react";
import { getProfiles, scoreSession, identifyTypist } from "@/lib/submit";
import { KeyEvent, SessionData } from "@/lib/types";
import { normaliseKey } from "@/lib/keyUtils";

const TARGET_MIN_KEYS = 40;

export default function VerifyTest({ onBack }: { onBack: () => void }) {
  const [profiles, setProfiles] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [mode, setMode] = useState<"verify" | "identify">("verify");
  
  // Selection states
  const [selectedProfile, setSelectedProfile] = useState<string>("");
  const [candidateIds, setCandidateIds] = useState<string[]>([]);
  
  // Typing states
  const [typed, setTyped] = useState("");
  const eventsRef = useRef<KeyEvent[]>([]);
  const [keyCount, setKeyCount] = useState(0);
  
  // Results states
  const [result, setResult] = useState<any>(null);
  const [scoring, setScoring] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    async function load() {
      const list = await getProfiles();
      setProfiles(list);
      setLoading(false);
      
      const trained = list.filter((p) => p.is_trained);
      if (trained.length > 0) {
        setSelectedProfile(trained[0].subject_id);
        setCandidateIds(trained.slice(0, 3).map((p) => p.subject_id));
      }
    }
    load();
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (["Shift", "Control", "Alt", "Meta", "CapsLock"].includes(e.key)) return;
    const { id, category } = normaliseKey(e.key);
    const ts = performance.now();
    eventsRef.current.push({
      key_id: id,
      key_category: category,
      press_ts: ts,
      release_ts: null,
      segment: "free",
    });
    setKeyCount(eventsRef.current.length);
  };

  const handleKeyUp = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (["Shift", "Control", "Alt", "Meta", "CapsLock"].includes(e.key)) return;
    const ts = performance.now();
    const { id } = normaliseKey(e.key);
    for (let i = eventsRef.current.length - 1; i >= 0; i--) {
      const ev = eventsRef.current[i];
      if (ev.key_id === id && ev.release_ts === null) {
        ev.release_ts = ts;
        break;
      }
    }
  };

  const buildSession = (): SessionData => {
    return {
      subject_id: "test-verif",
      collected_at: new Date().toISOString(),
      duration_ms: 0,
      keyboard: {
        events: eventsRef.current,
        pangram_text_length: 0,
        free_text_length: typed.length,
      },
      mouse: {
        passive_points: [],
        dot_trials: [],
        drag_trials: [],
      },
    };
  };

  const runAnalysis = async () => {
    if (eventsRef.current.length < TARGET_MIN_KEYS) return;
    setScoring(true);
    setErrorMsg("");
    setResult(null);

    const session = buildSession();

    try {
      if (mode === "verify") {
        if (!selectedProfile) throw new Error("Please select a profile to verify against.");
        const res = await scoreSession(selectedProfile, session);
        setResult(res);
      } else {
        if (candidateIds.length < 2) throw new Error("Please select at least 2 candidate profiles.");
        const res = await identifyTypist(candidateIds, session);
        setResult(res);
      }
    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || "Request failed. Make sure the backend models are trained.");
    } finally {
      setScoring(false);
    }
  };

  const resetTest = () => {
    setTyped("");
    eventsRef.current = [];
    setKeyCount(0);
    setResult(null);
    setErrorMsg("");
  };

  const toggleCandidate = (id: string) => {
    setCandidateIds((prev) => 
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center font-mono-tight text-sm text-muted">
        loading behaveguard profiles...
      </div>
    );
  }

  const trainedProfiles = profiles.filter((p) => p.is_trained);

  return (
    <div className="flex-1 px-6 py-8 overflow-y-auto">
      <div className="max-w-xl mx-auto fade-up">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <div className="font-mono-tight text-xs uppercase tracking-[0.3em] text-muted mb-2">verify & identify</div>
            <h2 className="text-2xl font-semibold">Biometric Sandbox</h2>
          </div>
          <button 
            onClick={onBack}
            className="font-mono-tight text-xs uppercase tracking-wider border border-border px-4 py-2 rounded-md hover:bg-surface-2 transition"
          >
            ← back
          </button>
        </div>

        {trainedProfiles.length === 0 ? (
          <div className="bg-surface border border-border rounded-xl p-8 text-center">
            <p className="text-sm text-muted mb-4">No trained biometric profiles exist in the system yet.</p>
            <p className="text-xs text-muted leading-relaxed mb-6">
              Go back and run a standard enrollment test to capture training data and fit the machine learning models.
            </p>
            <button 
              onClick={onBack}
              className="font-mono-tight text-xs uppercase tracking-wider bg-amber text-bg px-6 py-2.5 rounded-md hover:brightness-105 transition"
            >
              go back
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Mode Switcher */}
            <div className="grid grid-cols-2 gap-2 bg-surface-2 p-1 rounded-lg border border-border">
              <button
                onClick={() => { setMode("verify"); resetTest(); }}
                className={`py-2 text-xs font-mono-tight uppercase tracking-wider rounded-md transition ${mode === "verify" ? "bg-background text-cyan shadow-sm" : "text-muted hover:text-text"}`}
              >
                1:1 Verification
              </button>
              <button
                onClick={() => { setMode("identify"); resetTest(); }}
                className={`py-2 text-xs font-mono-tight uppercase tracking-wider rounded-md transition ${mode === "identify" ? "bg-background text-amber shadow-sm" : "text-muted hover:text-text"}`}
              >
                1:N Identification
              </button>
            </div>

            {/* Profile Selection */}
            {mode === "verify" ? (
              <div className="bg-surface border border-border rounded-xl p-5">
                <label className="block font-mono-tight text-xs uppercase tracking-wider text-muted mb-2">
                  Verify against profile:
                </label>
                <select
                  value={selectedProfile}
                  onChange={(e) => { setSelectedProfile(e.target.value); resetTest(); }}
                  className="w-full bg-background border border-border rounded-md px-3 py-2 text-sm text-text font-mono-tight focus:outline-none focus:border-cyan"
                >
                  {trainedProfiles.map((p) => (
                    <option key={p.subject_id} value={p.subject_id}>
                      {p.subject_id} (trained)
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div className="bg-surface border border-border rounded-xl p-5">
                <label className="block font-mono-tight text-xs uppercase tracking-wider text-muted mb-3">
                  Select candidate profiles (min 2):
                </label>
                <div className="grid grid-cols-2 gap-3">
                  {trainedProfiles.map((p) => {
                    const isChecked = candidateIds.includes(p.subject_id);
                    return (
                      <div 
                        key={p.subject_id} 
                        onClick={() => { toggleCandidate(p.subject_id); resetTest(); }}
                        className={`flex items-center gap-3 border rounded-lg p-3 cursor-pointer transition select-none ${isChecked ? "border-amber bg-amber/5 text-amber" : "border-border hover:border-text/30"}`}
                      >
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => {}} // handled by div onClick
                          className="accent-amber"
                        />
                        <span className="font-mono-tight text-sm">{p.subject_id}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Typing Sandbox */}
            <div className="bg-surface border border-border rounded-xl p-5 space-y-4">
              <div className="flex justify-between items-center text-xs font-mono-tight text-muted">
                <span>typing sandbox</span>
                <span className={keyCount >= TARGET_MIN_KEYS ? "text-green" : "text-amber"}>
                  {keyCount} / {TARGET_MIN_KEYS} keys
                </span>
              </div>

              <div className="text-xs text-muted bg-surface-2 border border-border rounded p-3 font-mono-tight leading-relaxed">
                <span className="text-text/40">Prompt Suggestion:</span> "the quick brown fox jumps over the lazy dog. pack my box with five dozen liquor jugs."
              </div>

              <input
                type="text"
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                onKeyDown={handleKeyDown}
                onKeyUp={handleKeyUp}
                disabled={scoring}
                placeholder="Type here to capture your biometric key rhythms..."
                className="w-full bg-background border border-border rounded-md px-4 py-3 text-sm focus:outline-none focus:border-text transition font-mono-tight"
              />

              <div className="flex gap-3 justify-end">
                <button
                  onClick={resetTest}
                  disabled={keyCount === 0 || scoring}
                  className="font-mono-tight text-xs uppercase tracking-wider border border-border px-4 py-2.5 rounded hover:bg-surface-2 transition disabled:opacity-50"
                >
                  reset
                </button>
                <button
                  onClick={runAnalysis}
                  disabled={keyCount < TARGET_MIN_KEYS || scoring}
                  className={`font-mono-tight text-xs uppercase tracking-wider text-bg px-6 py-2.5 rounded transition disabled:opacity-50 ${mode === "verify" ? "bg-cyan" : "bg-amber"}`}
                >
                  {scoring ? "analysing..." : mode === "verify" ? "verify signature" : "identify typist"}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {errorMsg && (
              <div className="bg-red/10 border border-red/30 text-red rounded-lg p-4 font-mono-tight text-xs leading-relaxed">
                ⚠️ {errorMsg}
              </div>
            )}

            {/* Results Rendering */}
            {result && (
              <div className="fade-up space-y-4">
                {mode === "verify" ? (
                  /* VERIFY RESULTS */
                  <div className="bg-surface border border-border rounded-xl p-5 space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="font-mono-tight text-xs uppercase tracking-widest text-muted">verification result</span>
                      <span className={`px-2.5 py-1 text-xs uppercase font-mono-tight tracking-wider rounded-md ${result.verdict === "legitimate" ? "bg-green/10 text-green" : result.verdict === "uncertain" ? "bg-amber/10 text-amber" : "bg-red/10 text-red"}`}>
                        {result.verdict}
                      </span>
                    </div>

                    <div className="bg-surface-2 border border-border rounded-lg p-4 text-center">
                      <div className="text-3xl font-semibold mb-1 font-mono-tight">
                        {Math.round((1 - result.anomaly_score) * 100)}%
                      </div>
                      <div className="text-[10px] uppercase tracking-widest text-muted font-mono-tight">biometric match rate</div>
                    </div>

                    <div className="space-y-3 pt-2">
                      <div className="text-xs font-mono-tight uppercase tracking-wider text-muted">model breakdown:</div>
                      
                      {/* SVM */}
                      <div className="flex justify-between items-center bg-surface-2 rounded p-2.5 border border-border text-xs font-mono-tight">
                        <span className="text-muted">One-Class SVM Baseline</span>
                        <span className={result.models.svm.verdict === "legitimate" ? "text-green" : "text-red"}>
                          {result.models.svm.verdict}
                        </span>
                      </div>

                      {/* LSTM */}
                      <div className="flex justify-between items-center bg-surface-2 rounded p-2.5 border border-border text-xs font-mono-tight">
                        <span className="text-muted">LSTM Sequence Autoencoder</span>
                        <span className={result.models.lstm.verdict === "legitimate" ? "text-green" : "text-red"}>
                          {result.models.lstm.verdict}
                        </span>
                      </div>

                      {/* TCN */}
                      <div className="flex justify-between items-center bg-surface-2 rounded p-2.5 border border-border text-xs font-mono-tight">
                        <span className="text-muted">TCN Dilation Autoencoder</span>
                        <span className={result.models.tcn.verdict === "legitimate" ? "text-green" : "text-red"}>
                          {result.models.tcn.verdict}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  /* IDENTIFY RESULTS */
                  <div className="bg-surface border border-border rounded-xl p-5 space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="font-mono-tight text-xs uppercase tracking-widest text-muted">typist identified</span>
                      <span className="font-mono-tight text-sm uppercase tracking-wider text-amber font-semibold">
                        {result.identified_subject_id}
                      </span>
                    </div>

                    <div className="space-y-3.5 pt-2">
                      <div className="text-xs font-mono-tight uppercase tracking-wider text-muted">confidence breakdown:</div>
                      {result.candidates.map((c: any) => (
                        <div key={c.subject_id} className="space-y-1">
                          <div className="flex justify-between text-xs font-mono-tight">
                            <span className="font-semibold">{c.subject_id}</span>
                            <span className="text-muted">{Math.round(c.confidence * 100)}% confidence ({c.verdict})</span>
                          </div>
                          <div className="h-2 bg-surface-2 rounded-full overflow-hidden border border-border">
                            <div 
                              className="h-full bg-amber rounded-full transition-all duration-500"
                              style={{ width: `${c.confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
