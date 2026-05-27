/// <reference types="vite/client" />

const base = (import.meta.env["VITE_API_URL"] as string | undefined) ?? "http://localhost:8000";

// ─── Types ────────────────────────────────────────────────────────────────────

export interface LorasResponse {
  steps: number[];
}

export interface CleanResponse {
  cleaned: string;
  original: string;
}

export interface GenerateRequest {
  prompt: string;
  lora_step?: number | null;
  size?: number;
  num_steps?: number;
  guidance_scale?: number;
  seed?: number;
}

export interface GenerateResult {
  imageUrl: string;
  cleanedPrompt: string;
  durationMs: number;
}

// ─── API Helpers ──────────────────────────────────────────────────────────────

async function handleResponse(resp: Response): Promise<void> {
  if (!resp.ok) {
    let message = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // ignore parse error
    }
    throw new Error(message);
  }
}

// ─── Endpoints ────────────────────────────────────────────────────────────────

/**
 * Fetches available LoRA checkpoint steps from the backend.
 * Also warms the inference container as a side effect.
 */
export async function getLoras(): Promise<LorasResponse> {
  const resp = await fetch(`${base}/loras`);
  await handleResponse(resp);
  return resp.json() as Promise<LorasResponse>;
}

/**
 * Sends a casual prompt to the Groq LLM for cleaning/structuring.
 */
export async function cleanPrompt(prompt: string): Promise<CleanResponse> {
  const resp = await fetch(`${base}/clean`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  await handleResponse(resp);
  return resp.json() as Promise<CleanResponse>;
}

/**
 * Generates an AI portrait image via FLUX.1-dev.
 *
 * Two-phase reveal:
 * 1. After `await fetch()` resolves (headers arrive ~2s), `onCleaned` is called
 *    immediately with the AI-cleaned prompt from the `x-cleaned-prompt` header.
 * 2. `await resp.blob()` then downloads the image binary (~25-30s more).
 *
 * @param req - Generation parameters
 * @param onCleaned - Callback invoked with cleaned prompt when headers arrive
 * @returns GenerateResult with blob object URL, cleaned prompt, and duration
 */
export async function generateImage(
  req: GenerateRequest,
  onCleaned: (prompt: string) => void
): Promise<GenerateResult> {
  const startMs = performance.now();

  const resp = await fetch(`${base}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });

  // Headers have arrived — extract cleaned prompt immediately (phase 1 → phase 2)
  const cleanedPrompt =
    resp.headers.get("x-cleaned-prompt") ?? req.prompt;
  onCleaned(cleanedPrompt);

  if (!resp.ok) {
    let message = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // ignore
    }
    throw new Error(message);
  }

  // Now await the full image body (phase 2 → done)
  const blob = await resp.blob();
  const imageUrl = URL.createObjectURL(blob);
  const durationMs = performance.now() - startMs;

  return { imageUrl, cleanedPrompt, durationMs };
}
