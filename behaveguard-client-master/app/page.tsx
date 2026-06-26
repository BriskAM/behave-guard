"use client";

import { useRef, useState } from "react";
import { Screen, SessionData, KeyEvent, DotTrial, DragTrial } from "@/lib/types";
import { usePassiveMouseCollector } from "@/lib/usePassiveMouse";
import { submitSession } from "@/lib/submit";
import StageRail from "@/components/StageRail";
import Landing from "@/components/Landing";
import Consent from "@/components/Consent";
import NameEntry from "@/components/NameEntry";
import KeyboardTest from "@/components/KeyboardTest";
import MouseDotTask from "@/components/MouseDotTask";
import MouseDragTask from "@/components/MouseDragTask";
import Analytics from "@/components/Analytics";
import VerifyTest from "@/components/VerifyTest";
import Done from "@/components/Done";

export default function Home() {
  const [screen, setScreen] = useState<Screen>("landing");
  const passivePoints = usePassiveMouseCollector(screen === "keyboard");
  const sessionStart = useRef(performance.now());

  const subjectId = useRef("");
  const keyboardData = useRef<{ events: KeyEvent[]; pangramLen: number; freeLen: number }>({
    events: [],
    pangramLen: 0,
    freeLen: 0,
  });
  const dotTrials = useRef<DotTrial[]>([]);
  const dragTrials = useRef<DragTrial[]>([]);
  const [sessionData, setSessionData] = useState<SessionData | null>(null);

  function buildSession(): SessionData {
    return {
      subject_id: subjectId.current,
      collected_at: new Date().toISOString(),
      duration_ms: performance.now() - sessionStart.current,
      keyboard: {
        events: keyboardData.current.events,
        pangram_text_length: keyboardData.current.pangramLen,
        free_text_length: keyboardData.current.freeLen,
      },
      mouse: {
        passive_points: passivePoints.current,
        dot_trials: dotTrials.current,
        drag_trials: dragTrials.current,
      },
    };
  }

  const showRail = screen !== "landing" && screen !== "done" && screen !== "verify";

  return (
    <div className="flex flex-col min-h-screen">
      {showRail && <StageRail current={screen} />}

      {screen === "landing" && (
        <Landing 
          onStart={() => setScreen("consent")} 
          onVerify={() => setScreen("verify")}
        />
      )}

      {screen === "consent" && <Consent onAgree={() => setScreen("name")} />}

      {screen === "name" && (
        <NameEntry
          onSubmit={(name) => {
            subjectId.current = name;
            setScreen("keyboard");
          }}
        />
      )}

      {screen === "keyboard" && (
        <KeyboardTest
          onComplete={(events, pangramLen, freeLen) => {
            keyboardData.current = { events, pangramLen, freeLen };
            setScreen("mouse-dot");
          }}
        />
      )}

      {screen === "mouse-dot" && (
        <MouseDotTask
          onComplete={(trials) => {
            dotTrials.current = trials;
            setScreen("mouse-drag");
          }}
        />
      )}

      {screen === "mouse-drag" && (
        <MouseDragTask
          onComplete={(trials) => {
            dragTrials.current = trials;
            const data = buildSession();
            setSessionData(data);
            submitSession(data); // fire immediately — don't wait for the user to click "finish"
            setScreen("analytics");
          }}
        />
      )}

      {screen === "analytics" && sessionData && (
        <Analytics data={sessionData} onFinish={() => setScreen("done")} />
      )}

      {screen === "verify" && (
        <VerifyTest onBack={() => setScreen("landing")} />
      )}

      {screen === "done" && <Done />}
    </div>
  );
}
