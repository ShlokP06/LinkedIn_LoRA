import { useState, useEffect } from "react";
import { motion } from "framer-motion";

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const scrollToGenerator = (e: React.MouseEvent) => {
    e.preventDefault();
    document.getElementById("generator")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const scrollTo = (id: string, offset = 0) => (e: React.MouseEvent) => {
    e.preventDefault();
    const el = document.getElementById(id);
    if (!el) return;
    window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY + offset, behavior: "smooth" });
  };

  return (
    <motion.nav
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="fixed top-0 left-0 right-0 z-40 flex items-center justify-between px-6 lg:px-10 h-16 transition-all duration-300"
      style={
        scrolled
          ? {
              background: "rgba(255,255,255,0.72)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              borderBottom: "1px solid rgba(255,255,255,0.5)",
              boxShadow: "0 4px 24px rgba(31,38,135,0.06)",
            }
          : {
              background: "transparent",
            }
      }
    >
      {/* Logo */}
      <a
        href="/"
        className="flex items-center gap-2 font-bold text-lg text-slate-800 hover:opacity-80 transition-opacity"
      >
        <span
          className="w-7 h-7 rounded-lg flex items-center justify-center text-sm font-black"
          style={{
            background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
            color: "white",
          }}
        >
          ✦
        </span>
        <span className="gradient-text-blue font-extrabold tracking-tight">
          Shlok.ai
        </span>
      </a>

      {/* Nav links */}
      <div className="hidden md:flex items-center gap-1">
        {([["Examples", "examples"], ["Features", "features"]] as [string, string][]).map(([label, id]) => (
          <a
            key={id}
            href={`#${id}`}
            onClick={scrollTo(id, id === "examples" ? -170 : 0)}
            className="px-4 py-2 rounded-xl text-sm font-medium text-slate-500 hover:text-slate-800 hover:bg-white/60 transition-all duration-200"
          >
            {label}
          </a>
        ))}
      </div>

      {/* CTA */}
      <motion.a
        href="#generator"
        onClick={scrollToGenerator}
        whileHover={{
          scale: 1.03,
          boxShadow: "0 8px 24px rgba(99,102,241,0.3)",
        }}
        whileTap={{ scale: 0.97 }}
        transition={{ type: "spring", stiffness: 350, damping: 25 }}
        className="hidden sm:flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold text-white"
        style={{
          background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
          boxShadow: "0 4px 14px rgba(99,102,241,0.2)",
        }}
      >
        Generate Portrait
        <span className="text-white/70 text-xs">↗</span>
      </motion.a>
    </motion.nav>
  );
}
