export type Screen =
  | "landing"
  | "consent"
  | "name"
  | "keyboard"
  | "mouse-dot"
  | "mouse-drag"
  | "analytics"
  | "verify"
  | "done";

export interface KeyEvent {
  key_id: string; // normalised key name, no content
  key_category: "alphanum" | "symbol" | "special" | "space";
  press_ts: number;
  release_ts: number | null;
  segment: "pangram" | "free";
}

export interface MousePoint {
  x: number;
  y: number;
  ts: number;
  dx: number;
  dy: number;
  pressure?: number;
}

export interface DotTrial {
  target_x: number;
  target_y: number;
  click_x: number;
  click_y: number;
  appeared_at: number;
  clicked_at: number;
  travel_time_ms: number;
  error_px: number;
  path: { x: number; y: number; ts: number }[];
}

export interface DragTrial {
  start_x: number;
  start_y: number;
  end_x: number;
  end_y: number;
  zone_x: number;
  zone_y: number;
  started_at: number;
  ended_at: number;
  duration_ms: number;
  success: boolean;
  path: { x: number; y: number; ts: number }[];
}

export interface SessionData {
  subject_id: string;
  collected_at: string;
  duration_ms: number;
  keyboard: {
    events: KeyEvent[];
    pangram_text_length: number;
    free_text_length: number;
  };
  mouse: {
    passive_points: MousePoint[];
    dot_trials: DotTrial[];
    drag_trials: DragTrial[];
  };
}
