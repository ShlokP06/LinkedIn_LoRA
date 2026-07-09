import { useState, useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { getLoras } from "./api";
import { Navbar } from "./components/Navbar";
import { Hero } from "./components/Hero";
import { Examples } from "./components/Examples";
import { Generator } from "./components/Generator";
import { Features } from "./components/Features";
import { Footer } from "./components/Footer";

function ToastNotification({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  useEffect(() => {
    const t = setTimeout(onDismiss, 5000);
    return () => clearTimeout(t);
  }, [onDismiss]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 8, scale: 0.95 }}
      transition={{ type: "spring", stiffness: 340, damping: 28 }}
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-5 py-3.5 rounded-2xl text-sm font-medium max-w-sm shadow-xl"
      style={{
        background: "rgba(254,202,202,0.85)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        border: "1px solid rgba(252,165,165,0.5)",
        color: "#991b1b",
        boxShadow: "0 16px 40px rgba(239,68,68,0.15)",
      }}
    >
      <span className="text-red-400">⚠</span>
      <span className="flex-1">{message}</span>
      <button
        onClick={onDismiss}
        className="text-red-400/60 hover:text-red-400 transition-colors text-lg leading-none"
      >
        ×
      </button>
    </motion.div>
  );
}

export default function App() {
  const availableSteps: number[] = [1750, 2500, 2750, 3000];
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Warm the Modal GPU container as soon as the page loads — cold start takes
  // 20-60s, so firing this early hides most of it before the first generate.
  useEffect(() => {
    void getLoras().catch(() => {});
  }, []);

  const dismissToast = () => setToastMessage(null);

  return (
    <div className="relative min-h-screen aurora-bg grain-overlay">
      {/* Global background base */}
      <div
        className="fixed inset-0 pointer-events-none -z-10"
        style={{
          background:
            "linear-gradient(160deg, #f8f7ff 0%, #fdf4ff 30%, #f0f9ff 60%, #fff7ed 100%)",
        }}
      />

      <Navbar />

      <main>
        <Hero />

        {/* Divider between sections */}
        <div className="h-px mx-10 lg:mx-24 bg-gradient-to-r from-transparent via-slate-200/70 to-transparent" />

        <Examples />

        <div className="h-px mx-10 lg:mx-24 bg-gradient-to-r from-transparent via-slate-200/70 to-transparent" />

        <Generator availableSteps={availableSteps} />

        <div className="h-px mx-10 lg:mx-24 bg-gradient-to-r from-transparent via-slate-200/70 to-transparent" />

        <Features />
      </main>

      <Footer />

      {/* Toast notifications */}
      <AnimatePresence>
        {toastMessage && (
          <ToastNotification
            key={toastMessage}
            message={toastMessage}
            onDismiss={dismissToast}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
