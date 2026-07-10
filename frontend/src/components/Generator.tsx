import { useState, useRef, useCallback } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";
import {
  ChevronDown,
  Wand2,
  Loader2,
  Sparkles,
} from "lucide-react";
import { PillSelector } from "./PillSelector";
import { GeneratorOutput, type GenerationPhase } from "./GeneratorOutput";
import { generateImage, type GenerateRequest } from "../api";

interface GeneratorProps {
  availableSteps: number[];
}

const SIZE_OPTIONS = [
  { value: 512, label: "512" },
  { value: 768, label: "768" },
  { value: 1024, label: "1024" },
  { value: 1536, label: "1536" },
];

// Magnetic button hook
function useMagneticButton() {
  const ref = useRef<HTMLButtonElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const springX = useSpring(x, { stiffness: 140, damping: 22 });
  const springY = useSpring(y, { stiffness: 140, damping: 22 });

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLButtonElement>) => {
    const btn = ref.current;
    if (!btn) return;
    const rect = btn.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    x.set((e.clientX - cx) * 0.18);
    y.set((e.clientY - cy) * 0.18);
  }, [x, y]);

  const handleMouseLeave = useCallback(() => {
    x.set(0);
    y.set(0);
  }, [x, y]);

  return { ref, springX, springY, handleMouseMove, handleMouseLeave };
}

export function Generator({ availableSteps }: GeneratorProps) {
  // Form state
  const [prompt, setPrompt] = useState("");
  const [size, setSize] = useState(768);
  const [seed, setSeed] = useState(42);
  const [numSteps, setNumSteps] = useState(22);
  const [guidanceScale, setGuidanceScale] = useState(4.0);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [enhance, setEnhance] = useState(true);

  // Output state
  const [phase, setPhase] = useState<GenerationPhase>("idle");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [cleanedPrompt, setCleanedPrompt] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [meta, setMeta] = useState<{
    durationMs: number;
    seed: number;
    size: number;
  } | null>(null);

  const magnetic = useMagneticButton();
  const generateBtnRef = useRef<HTMLButtonElement>(null);
  const advancedContentRef = useRef<HTMLDivElement>(null);
  const sectionRef = useRef<HTMLElement>(null);
  const isGenerating = phase === "phase1" || phase === "phase2";

  const handleAdvancedToggle = () => {
    const opening = !advancedOpen;
    setAdvancedOpen(opening);

    // Use rAF so the state flip + animation start on the same frame,
    // then scroll in the very next frame — fully parallel visually.
    requestAnimationFrame(() => {
      if (opening) {
        const expandedHeight = advancedContentRef.current?.scrollHeight ?? 280;
        const btn = generateBtnRef.current;
        if (!btn) return;
        const rect = btn.getBoundingClientRect();
        const targetScroll =
          window.scrollY + rect.bottom + expandedHeight - window.innerHeight + 65;
        window.scrollTo({ top: targetScroll, behavior: "smooth" });
      } else {
        const section = sectionRef.current;
        if (!section) return;
        const top = window.scrollY + section.getBoundingClientRect().top - 72;
        window.scrollTo({ top, behavior: "smooth" });
      }
    });
  };

  const handleGenerate = async () => {
    if (!prompt.trim() || isGenerating) return;

    setPhase("phase1");
    setCleanedPrompt(null);
    setImageUrl(null);
    setErrorMessage(null);
    setMeta(null);

    const req: GenerateRequest = {
      prompt: prompt.trim(),
      lora_step: availableSteps.includes(2750) ? 2750 : availableSteps[0] ?? 2750,
      size,
      num_steps: numSteps,
      guidance_scale: guidanceScale,
      seed,
      enhance,
    };

    try {
      const result = await generateImage(req, (cleaned) => {
        // With enhancement off the "cleaned" prompt is just the raw input —
        // no point showing it back to the user in the AI-Enhanced Prompt card.
        setCleanedPrompt(enhance ? cleaned : null);
        setPhase("phase2");
      });

      setImageUrl(result.imageUrl);
      setMeta({
        durationMs: result.durationMs,
        seed,
        size,
      });
      setPhase("done");
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Generation failed. Please try again.";
      setErrorMessage(message);
      setPhase("error");
    }
  };

  const handleRegenerate = () => {
    void handleGenerate();
  };

  return (
    <section ref={sectionRef} id="generator" className="relative pt-8 pb-24 px-4 scroll-mt-16">
      <div className="max-w-7xl mx-auto">
        {/* Section label */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.65, ease: [0.16, 1, 0.3, 1] }}
          className="text-center mb-8"
        >
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-medium mb-5 badge-shimmer"
            style={{
              background: "linear-gradient(135deg, rgba(196,181,253,0.35), rgba(186,230,253,0.3))",
              border: "1px solid rgba(139,92,246,0.2)",
              color: "#7c3aed",
            }}
          >
            <Wand2 className="w-3.5 h-3.5" />
            AI Portrait Generator
          </div>
          <h2 className="text-4xl font-bold text-slate-800 mb-3">
            Create your{" "}
            <span className="gradient-text">perfect portrait</span>
          </h2>
          <p className="text-slate-500 max-w-xl mx-auto text-lg">
            Describe your vision. Our AI handles the rest.
          </p>
        </motion.div>

        {/* Main generator card */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="relative rounded-3xl overflow-hidden"
          style={{
            background: "rgba(255,255,255,0.65)",
            backdropFilter: "blur(40px)",
            WebkitBackdropFilter: "blur(40px)",
            border: "1px solid rgba(255,255,255,0.5)",
            boxShadow:
              "0 32px 80px rgba(31,38,135,0.12), 0 12px 32px rgba(31,38,135,0.07), inset 0 1px 0 rgba(255,255,255,0.8)",
          }}
        >
          <div className="grid lg:grid-cols-2 gap-0 divide-x divide-white/40">
            {/* ─── Left Panel: Inputs ─────────────────────────────────── */}
            <div className="p-7 lg:p-8 space-y-5">
              {/* Prompt */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm font-semibold text-slate-700">
                    Describe your portrait
                  </label>
                  {/* AI Enhance toggle — off sends the prompt to FLUX unmodified */}
                  <button
                    type="button"
                    role="switch"
                    aria-checked={enhance}
                    onClick={() => setEnhance((v) => !v)}
                    disabled={isGenerating}
                    title={
                      enhance
                        ? "Your description is rewritten into an optimized portrait prompt"
                        : "Your description is sent to FLUX exactly as written"
                    }
                    className="flex items-center gap-2 text-xs font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{ color: enhance ? "#7c3aed" : "#94a3b8" }}
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    AI Enhance
                    <span
                      className="relative inline-flex w-9 h-5 rounded-full transition-colors duration-200"
                      style={{
                        background: enhance
                          ? "linear-gradient(135deg, #8b5cf6, #6366f1)"
                          : "rgba(203,213,225,0.9)",
                      }}
                    >
                      <motion.span
                        layout
                        transition={{ type: "spring", stiffness: 500, damping: 32 }}
                        className="absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm"
                        style={{ left: enhance ? "calc(100% - 18px)" : "2px" }}
                      />
                    </span>
                  </button>
                </div>
                <div className="relative">
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder="e.g. Professional headshot, smart casual, soft studio lighting, confident expression, looking into camera…"
                    rows={4}
                    maxLength={500}
                    disabled={isGenerating}
                    className="w-full px-4 py-3.5 rounded-2xl text-sm text-slate-700 placeholder:text-slate-300 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      background: "rgba(255,255,255,0.8)",
                      border: "1.5px solid rgba(203,213,225,0.8)",
                      outline: "none",
                      boxShadow: "0 2px 8px rgba(31,38,135,0.04)",
                    }}
                    onFocus={(e) => {
                      e.currentTarget.style.border = "1.5px solid rgba(139,92,246,0.5)";
                      e.currentTarget.style.boxShadow =
                        "0 0 0 3px rgba(139,92,246,0.12), 0 2px 8px rgba(31,38,135,0.04)";
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.border = "1.5px solid rgba(203,213,225,0.8)";
                      e.currentTarget.style.boxShadow = "0 2px 8px rgba(31,38,135,0.04)";
                    }}
                  />
                  <div className="flex justify-end mt-1.5 px-1">
                    <span
                      className="text-xs tabular-nums"
                      style={{
                        color: prompt.length > 450 ? "#ef4444" : "#94a3b8",
                      }}
                    >
                      {prompt.length}/500
                    </span>
                  </div>
                </div>
              </div>

              {/* Size */}
              <div className="space-y-3">
                <label className="block text-sm font-semibold text-slate-700">
                  Output Size
                </label>
                <PillSelector
                  options={SIZE_OPTIONS}
                  value={size}
                  onChange={setSize}
                  layoutId="size-pill"
                />
              </div>

              {/* Separator */}
              <div className="h-px bg-gradient-to-r from-transparent via-slate-200/80 to-transparent" />

              {/* Advanced Options */}
              <div>
                <button
                  onClick={handleAdvancedToggle}
                  className="flex items-center gap-2 text-sm font-semibold text-slate-500 hover:text-slate-700 transition-colors group"
                >
                  <motion.span
                    animate={{ rotate: advancedOpen ? 180 : 0 }}
                    transition={{ type: "spring", stiffness: 180, damping: 28 }}
                  >
                    <ChevronDown className="w-4 h-4" />
                  </motion.span>
                  Advanced Options
                </button>

                <motion.div
                  animate={{ height: advancedOpen ? "auto" : 0, opacity: advancedOpen ? 1 : 0 }}
                  initial={false}
                  transition={{ duration: 0.45, ease: [0.25, 0.46, 0.45, 0.94] }}
                  className="overflow-hidden"
                >
                  <div ref={advancedContentRef} className="pt-5 space-y-5">
                        {/* Seed */}
                        <div className="space-y-2">
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 uppercase tracking-wider">
                            Seed
                            <span className="font-mono text-slate-400 normal-case tracking-normal text-sm">
                              {seed}
                            </span>
                          </label>
                          <input
                            type="number"
                            value={seed}
                            onChange={(e) => setSeed(Number(e.target.value))}
                            min={0}
                            max={2147483647}
                            disabled={isGenerating}
                            className="w-full px-3.5 py-2.5 rounded-xl text-sm text-slate-700 font-mono disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                            style={{
                              background: "rgba(255,255,255,0.7)",
                              border: "1.5px solid rgba(203,213,225,0.7)",
                              outline: "none",
                            }}
                            onFocus={(e) => {
                              e.currentTarget.style.border = "1.5px solid rgba(139,92,246,0.4)";
                              e.currentTarget.style.boxShadow =
                                "0 0 0 3px rgba(139,92,246,0.1)";
                            }}
                            onBlur={(e) => {
                              e.currentTarget.style.border = "1.5px solid rgba(203,213,225,0.7)";
                              e.currentTarget.style.boxShadow = "none";
                            }}
                          />
                        </div>

                        {/* Inference Steps */}
                        <div className="space-y-2">
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 uppercase tracking-wider">
                            Inference Steps
                            <span className="font-mono text-violet-500 normal-case tracking-normal text-sm">
                              {numSteps}
                            </span>
                          </label>
                          <input
                            type="range"
                            min={20}
                            max={50}
                            step={1}
                            value={numSteps}
                            onChange={(e) => setNumSteps(Number(e.target.value))}
                            disabled={isGenerating}
                            className="w-full disabled:opacity-50 disabled:cursor-not-allowed"
                          />
                          <div className="flex justify-between text-xs text-slate-400">
                            <span>20 (fast)</span>
                            <span>50 (quality)</span>
                          </div>
                        </div>

                        {/* Guidance Scale */}
                        <div className="space-y-2">
                          <label className="flex items-center justify-between text-xs font-semibold text-slate-600 uppercase tracking-wider">
                            Guidance Scale
                            <span className="font-mono text-violet-500 normal-case tracking-normal text-sm">
                              {guidanceScale.toFixed(1)}
                            </span>
                          </label>
                          <input
                            type="range"
                            min={2}
                            max={5}
                            step={0.1}
                            value={guidanceScale}
                            onChange={(e) => setGuidanceScale(Number(e.target.value))}
                            disabled={isGenerating}
                            className="w-full disabled:opacity-50 disabled:cursor-not-allowed"
                          />
                          <div className="flex justify-between text-xs text-slate-400">
                            <span>2.0 (natural)</span>
                            <span>5.0 (strict)</span>
                          </div>
                        </div>
                  </div>
                </motion.div>
              </div>

              {/* Generate Button */}
              <motion.button
                ref={(el) => {
                  (magnetic.ref as React.MutableRefObject<HTMLButtonElement | null>).current = el;
                  generateBtnRef.current = el;
                }}
                onClick={() => void handleGenerate()}
                disabled={isGenerating || !prompt.trim()}
                onMouseMove={magnetic.handleMouseMove}
                onMouseLeave={magnetic.handleMouseLeave}
                style={{
                  x: magnetic.springX,
                  y: magnetic.springY,
                  background:
                    "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)",
                  boxShadow:
                    "0 8px 24px rgba(99,102,241,0.25), 0 2px 8px rgba(139,92,246,0.15)",
                }}
                whileHover={
                  !isGenerating && prompt.trim()
                    ? { scale: 1.02, boxShadow: "0 16px 40px rgba(99,102,241,0.35), 0 4px 12px rgba(139,92,246,0.25)" }
                    : {}
                }
                whileTap={
                  !isGenerating && prompt.trim() ? { scale: 0.98 } : {}
                }
                transition={{ type: "spring", stiffness: 220, damping: 28 }}
                className="w-full py-4 rounded-2xl font-semibold text-white text-base relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {/* Shimmer overlay on hover */}
                <span
                  className="absolute inset-0 opacity-0 hover:opacity-100 transition-opacity duration-300"
                  style={{
                    background:
                      "linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.15) 50%, transparent 70%)",
                  }}
                />

                <span className="relative z-10 flex items-center justify-center gap-2.5">
                  {isGenerating ? (
                    <>
                      <Loader2 className="w-4.5 h-4.5 animate-spin" />
                      Generating portrait…
                    </>
                  ) : (
                    <>
                      <Wand2 className="w-4.5 h-4.5" />
                      Generate Portrait
                    </>
                  )}
                </span>
              </motion.button>
            </div>

            {/* ─── Right Panel: Output ────────────────────────────────── */}
            <div className="p-7 lg:p-8 flex flex-col">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">
                  Output
                </h3>
                {phase === "done" && (
                  <motion.span
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full"
                    style={{
                      background: "rgba(187,247,208,0.5)",
                      color: "#15803d",
                      border: "1px solid rgba(134,239,172,0.4)",
                    }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
                    Ready
                  </motion.span>
                )}
              </div>

              <div className="flex-1 flex flex-col min-h-0">
                <GeneratorOutput
                  phase={phase}
                  imageUrl={imageUrl}
                  cleanedPrompt={cleanedPrompt}
                  meta={meta}
                  errorMessage={errorMessage}
                  onRegenerate={handleRegenerate}
                  enhance={enhance}
                />
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
