import { SessionData } from "./types";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function submitSession(data: SessionData): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error("Submit failed", err);
    return null;
  }
}

export async function getProfiles(): Promise<any[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/profiles`);
    if (!response.ok) return [];
    return await response.json();
  } catch (err) {
    console.error("Failed to load profiles", err);
    return [];
  }
}

export async function triggerTraining(subjectId: string): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject_id: subjectId }),
    });
    return await response.json();
  } catch (err) {
    console.error("Trigger training failed", err);
    return null;
  }
}

export async function getTrainingStatus(subjectId: string): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/profiles/${subjectId}/status`);
    if (!response.ok) return null;
    return await response.json();
  } catch (err) {
    console.error("Failed to get training status", err);
    return null;
  }
}

export async function scoreSession(subjectId: string, session: SessionData): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ subject_id: subjectId, session }),
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error("Scoring failed", err);
    throw err;
  }
}

export async function identifyTypist(candidateIds: string[], session: SessionData): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/identify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate_ids: candidateIds, session }),
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP error! Status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    console.error("Identification failed", err);
    throw err;
  }
}

