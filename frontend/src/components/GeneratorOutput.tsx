import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Download,
  Maximize2,
  RotateCcw,
  Sparkles,
  ImageIcon,
  Loader2,
} from "lucide-react";

export type GenerationPhase = "idle" | "phase1" | "phase2" | "done" | "error";

interface GenerationMeta {
  durationMs: number;
  seed: number;
  size: number;
}

interface GeneratorOutputProps {
  phase: GenerationPhase;
  imageUrl: string | null;
  cleanedPrompt: string | null;
  meta: GenerationMeta | null;
  errorMessage: string | null;
  onRegenerate: () => void;
  /** Whether the Groq cleaning pipeline is active — adjusts status copy */
  enhance?: boolean;
}

function SkeletonBlock({ className }: { className?: string }) {
  return (
    <div
      className={`relative overflow-hidden rounded-xl bg-gradient-to-r from-slate-100 via-white to-slate-100 ${className ?? ""}`}
    >
      <div className="shimmer absolute inset-0" />
    </div>
  );
}

function IdleState() {
  return (
    <motion.div
      key="idle"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="relative w-full h-full min-h-[420px] flex-1 rounded-2xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center gap-5 overflow-hidden"
    >
        {/* Subtle gradient background */}
        <div className="absolute inset-0 bg-gradient-to-br from-violet-50/60 via-sky-50/40 to-peach-50/40 rounded-2xl" />

        {/* Floating particles */}
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1.5 h-1.5 rounded-full"
            style={{
              background: `hsl(${240 + i * 20}, 70%, 75%)`,
              left: `${15 + i * 14}%`,
              top: `${20 + (i % 3) * 25}%`,
            }}
            animate={{
              y: [0, -10, 0],
              opacity: [0.4, 0.8, 0.4],
            }}
            transition={{
              duration: 2.5 + i * 0.4,
              repeat: Infinity,
              ease: "easeInOut",
              delay: i * 0.3,
            }}
          />
        ))}

        <motion.div
          animate={{ y: [0, -6, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          className="relative z-10 w-16 h-16 rounded-2xl glass flex items-center justify-center shadow-glass"
          style={{
            background:
              "linear-gradient(135deg, rgba(196,181,253,0.4), rgba(186,230,253,0.4))",
          }}
        >
          <ImageIcon className="w-7 h-7 text-violet-400" />
        </motion.div>

        <div className="relative z-10 text-center space-y-2">
          <p className="text-slate-500 font-medium text-sm">
            Your portrait will appear here
          </p>
          <p className="text-slate-400 text-xs">
            Fill in the form and hit Generate
          </p>
        </div>
    </motion.div>
  );
}

function GeneratingState({
  phase,
  cleanedPrompt,
  enhance,
}: {
  phase: "phase1" | "phase2";
  cleanedPrompt: string | null;
  enhance: boolean;
}) {
  return (
    <motion.div
      key="generating"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col gap-4 min-h-[420px]"
    >
      {/* Status bar */}
      <div className="flex items-center gap-2 px-4 py-2.5 rounded-xl glass-subtle">
        <Loader2 className="w-4 h-4 text-violet-500 animate-spin flex-shrink-0" />
        <span className="text-sm font-medium text-slate-600">
          {phase === "phase1"
            ? enhance
              ? "Crafting your prompt with Groq…"
              : "Sending your prompt to FLUX…"
            : "Generating your portrait with FLUX.1-dev…"}
        </span>
      </div>

      {/* Cleaned prompt reveal */}
      <AnimatePresence mode="wait">
        {phase === "phase2" && cleanedPrompt && (
          <motion.div
            key="cleaned-prompt"
            initial={{ opacity: 0, y: 8, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            className="relative rounded-2xl p-4 overflow-hidden"
            style={{
              background:
                "linear-gradient(135deg, rgba(196,181,253,0.2), rgba(186,230,253,0.15))",
              border: "1px solid rgba(139,92,246,0.2)",
            }}
          >
            <div className="flex items-start gap-2.5">
              <div className="flex-shrink-0 mt-0.5">
                <Sparkles className="w-4 h-4 text-violet-500" />
              </div>
              <div>
                <p className="text-xs font-semibold text-violet-600 mb-1.5 uppercase tracking-wider">
                  AI-Enhanced Prompt
                </p>
                <p className="text-sm text-slate-600 leading-relaxed font-medium">
                  "{cleanedPrompt}"
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Image skeleton with rotating border */}
      <div className="relative rotating-border rounded-2xl flex-1 min-h-[320px]">
        <div className="relative rounded-2xl overflow-hidden h-full min-h-[320px] glass-subtle flex flex-col items-center justify-center gap-4">
          {/* Skeleton blocks */}
          <div className="absolute inset-0 flex flex-col gap-3 p-4">
            <SkeletonBlock className="h-1/2" />
            <div className="flex gap-3 flex-1">
              <SkeletonBlock className="w-2/3" />
              <SkeletonBlock className="flex-1" />
            </div>
          </div>

          {/* Center overlay text */}
          <div className="relative z-10 text-center">
            <motion.div
              animate={{ scale: [1, 1.08, 1], opacity: [0.6, 1, 0.6] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            >
              <div className="w-12 h-12 mx-auto mb-3 rounded-xl flex items-center justify-center"
                style={{
                  background: "linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15))",
                }}>
                <Sparkles className="w-5 h-5 text-violet-500" />
              </div>
            </motion.div>
            <p className="text-sm font-semibold text-slate-500">
              {phase === "phase1" ? "Preparing…" : "Rendering portrait…"}
            </p>
            <p className="text-xs text-slate-400 mt-1">
              This takes about 25–30 seconds
            </p>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function DoneState({
  imageUrl,
  cleanedPrompt,
  meta,
  onRegenerate,
}: {
  imageUrl: string;
  cleanedPrompt: string | null;
  meta: GenerationMeta | null;
  onRegenerate: () => void;
}) {
  const [isHovering, setIsHovering] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = imageUrl;
    a.download = `portrait-seed${meta?.seed ?? 0}.png`;
    a.click();
  };

  const durationSec =
    meta ? (meta.durationMs / 1000).toFixed(1) : null;

  return (
    <motion.div
      key="done"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col gap-4"
    >
      {/* Cleaned prompt */}
      {cleanedPrompt && (
        <motion.div
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, type: "spring", stiffness: 300, damping: 28 }}
          className="relative rounded-2xl p-4 overflow-hidden"
          style={{
            background:
              "linear-gradient(135deg, rgba(196,181,253,0.2), rgba(186,230,253,0.15))",
            border: "1px solid rgba(139,92,246,0.2)",
          }}
        >
          <div className="flex items-start gap-2.5">
            <Sparkles className="w-4 h-4 text-violet-500 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-violet-600 mb-1 uppercase tracking-wider">
                AI-Enhanced Prompt
              </p>
              <p className="text-sm text-slate-600 leading-relaxed">
                "{cleanedPrompt}"
              </p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Image */}
      <div
        className="relative rounded-2xl overflow-hidden group"
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
      >
        <motion.img
          src={imageUrl}
          alt="Generated portrait"
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{
            duration: 0.6,
            ease: [0.16, 1, 0.3, 1],
          }}
          className="w-full rounded-2xl object-cover"
          style={{
            boxShadow:
              "0 24px 64px rgba(31, 38, 135, 0.2), 0 8px 24px rgba(31, 38, 135, 0.1)",
          }}
        />

        {/* Hover overlay with controls */}
        <AnimatePresence>
          {isHovering && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              className="absolute inset-0 rounded-2xl flex items-end justify-center pb-4"
              style={{
                background:
                  "linear-gradient(to top, rgba(0,0,0,0.5) 0%, transparent 60%)",
              }}
            >
              <div className="flex items-center gap-2">
                <motion.button
                  whileHover={{ scale: 1.08 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleDownload}
                  className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-sm font-semibold text-white"
                  style={{
                    background: "rgba(255,255,255,0.2)",
                    backdropFilter: "blur(12px)",
                    border: "1px solid rgba(255,255,255,0.3)",
                  }}
                >
                  <Download className="w-3.5 h-3.5" />
                  Download
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.08 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsFullscreen(true)}
                  className="p-2 rounded-xl text-white"
                  style={{
                    background: "rgba(255,255,255,0.2)",
                    backdropFilter: "blur(12px)",
                    border: "1px solid rgba(255,255,255,0.3)",
                  }}
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </motion.button>

                <motion.button
                  whileHover={{ scale: 1.08 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={onRegenerate}
                  className="p-2 rounded-xl text-white"
                  style={{
                    background: "rgba(255,255,255,0.2)",
                    backdropFilter: "blur(12px)",
                    border: "1px solid rgba(255,255,255,0.3)",
                  }}
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </motion.button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Metadata row */}
      {meta && (
        <motion.div
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="flex flex-wrap items-center gap-2 text-xs text-slate-400"
        >
          {durationSec && (
            <span className="px-2 py-1 rounded-lg bg-slate-100 text-slate-500 font-medium">
              {durationSec}s
            </span>
          )}
          <span className="px-2 py-1 rounded-lg bg-sky-50 text-sky-500 font-medium">
            Seed {meta.seed}
          </span>
          <span className="px-2 py-1 rounded-lg bg-slate-100 text-slate-500 font-medium">
            {meta.size}×{meta.size}
          </span>

          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={handleDownload}
            className="ml-auto flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold text-violet-600 border border-violet-200 hover:bg-violet-50 transition-colors"
          >
            <Download className="w-3 h-3" />
            Save PNG
          </motion.button>
        </motion.div>
      )}

      {/* Fullscreen modal */}
      <AnimatePresence>
        {isFullscreen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsFullscreen(false)}
            className="fixed inset-0 z-50 flex items-center justify-center p-6"
            style={{ background: "rgba(0,0,0,0.85)", backdropFilter: "blur(8px)" }}
          >
            <motion.img
              src={imageUrl}
              alt="Portrait fullscreen"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
              className="max-w-full max-h-full rounded-2xl object-contain"
              style={{ boxShadow: "0 32px 80px rgba(0,0,0,0.6)" }}
              onClick={(e) => e.stopPropagation()}
            />
            <button
              onClick={() => setIsFullscreen(false)}
              className="absolute top-6 right-6 w-10 h-10 rounded-full flex items-center justify-center text-white/80 hover:text-white transition-colors"
              style={{ background: "rgba(255,255,255,0.1)", backdropFilter: "blur(8px)" }}
            >
              ✕
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <motion.div
      key="error"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0 }}
      className="flex flex-col items-center justify-center min-h-[420px] gap-4"
    >
      <div
        className="w-full rounded-2xl p-6 text-center"
        style={{
          background:
            "linear-gradient(135deg, rgba(254,205,211,0.3), rgba(253,164,175,0.15))",
          border: "1px solid rgba(251,113,133,0.25)",
        }}
      >
        <div className="w-12 h-12 mx-auto mb-4 rounded-xl flex items-center justify-center text-2xl"
          style={{ background: "rgba(254,205,211,0.5)" }}>
          ✕
        </div>
        <p className="font-semibold text-rose-600 mb-1">Generation failed</p>
        <p className="text-sm text-rose-500/80 max-w-xs mx-auto">{message}</p>
      </div>
    </motion.div>
  );
}

export function GeneratorOutput({
  phase,
  imageUrl,
  cleanedPrompt,
  meta,
  errorMessage,
  onRegenerate,
  enhance = true,
}: GeneratorOutputProps) {
  const prevUrlRef = useRef<string | null>(null);

  // Revoke old blob URLs to prevent memory leaks
  useEffect(() => {
    if (imageUrl && imageUrl !== prevUrlRef.current) {
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
      }
      prevUrlRef.current = imageUrl;
    }
  }, [imageUrl]);

  useEffect(() => {
    return () => {
      if (prevUrlRef.current) {
        URL.revokeObjectURL(prevUrlRef.current);
      }
    };
  }, []);

  return (
    <div className="flex-1 flex flex-col h-full">
    <AnimatePresence mode="wait">
      {phase === "idle" && <IdleState key="idle" />}

      {(phase === "phase1" || phase === "phase2") && (
        <GeneratingState
          key="generating"
          phase={phase}
          cleanedPrompt={cleanedPrompt}
          enhance={enhance}
        />
      )}

      {phase === "done" && imageUrl && (
        <DoneState
          key="done"
          imageUrl={imageUrl}
          cleanedPrompt={cleanedPrompt}
          meta={meta}
          onRegenerate={onRegenerate}
        />
      )}

      {phase === "error" && (
        <ErrorState
          key="error"
          message={errorMessage ?? "An unknown error occurred."}
        />
      )}
    </AnimatePresence>
    </div>
  );
}
