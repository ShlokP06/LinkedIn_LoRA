import { motion } from "framer-motion";
import { Sparkles, ArrowRight } from "lucide-react";

const GRADIENT_VIOLET = "linear-gradient(135deg, #c4b5fd 0%, #818cf8 60%, #6366f1 100%)";
const GRADIENT_SKY = "linear-gradient(135deg, #bae6fd 0%, #7dd3fc 50%, #38bdf8 100%)";
const GRADIENT_PEACH = "linear-gradient(135deg, #fed7aa 0%, #fb923c 60%, #f97316 100%)";

const FLOAT_0 = { y: [0, -16, 0] as number[], duration: 4, delay: 0, rotate: -3 };
const FLOAT_1 = { y: [0, -12, 0] as number[], duration: 5, delay: 0.6, rotate: 3 };
const FLOAT_2 = { y: [0, -20, 0] as number[], duration: 6, delay: 1.2, rotate: -1.5 };

interface FloatConfig {
  y: number[];
  duration: number;
  delay: number;
  rotate: number;
}

function FloatingCard({
  gradient,
  float,
  depth,
  label,
}: {
  gradient: string;
  float: FloatConfig;
  depth: number;
  label: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 20 }}
      animate={{
        opacity: 1,
        scale: 1,
        y: float.y,
      }}
      transition={{
        opacity: { duration: 0.6, delay: 0.3 + depth * 0.15 },
        scale: { duration: 0.6, delay: 0.3 + depth * 0.15 },
        y: {
          repeat: Infinity,
          duration: float.duration,
          ease: "easeInOut",
          delay: float.delay,
        },
      }}
      className="absolute rounded-2xl overflow-hidden"
      style={{
        background: gradient,
        boxShadow:
          "0 24px 64px rgba(31,38,135,0.18), 0 8px 24px rgba(31,38,135,0.1)",
        transform: `rotate(${float.rotate}deg)`,
        zIndex: 3 - depth,
      }}
    >
      {/* Inner card content */}
      <div className="relative w-full h-full flex flex-col items-start justify-end p-3.5">
        {/* Frosted glass overlay at bottom */}
        <div
          className="absolute bottom-0 left-0 right-0 px-3.5 py-2.5 rounded-b-2xl"
          style={{
            background: "rgba(255,255,255,0.18)",
            backdropFilter: "blur(12px)",
          }}
        >
          <p className="text-white/90 text-xs font-semibold truncate">{label}</p>
          <div className="flex items-center gap-1 mt-0.5">
            <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
            <span className="text-white/60 text-[10px]">Shl0k · LinkedIn</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

const containerVariants = {
  hidden: {},
  show: {
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 24 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: [0.16, 1, 0.3, 1] },
  },
};

export function Hero() {
  const scrollToGenerator = (e: React.MouseEvent) => {
    e.preventDefault();
    document.getElementById("generator")?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <section className="relative min-h-[80vh] flex items-center pt-16 overflow-hidden">
      <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-10 w-full py-10 lg:py-16">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* ─── Left: Copy ─────────────────────────────────────────── */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="show"
            className="space-y-7"
          >
            {/* Badge */}
            <motion.div variants={itemVariants}>
              <div
                className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-semibold badge-shimmer"
                style={{
                  background:
                    "linear-gradient(135deg, rgba(196,181,253,0.4), rgba(186,230,253,0.35))",
                  border: "1px solid rgba(139,92,246,0.25)",
                  color: "#6d28d9",
                }}
              >
                <Sparkles className="w-3.5 h-3.5" />
                Powered by FLUX.1-dev ✦
              </div>
            </motion.div>

            {/* Headline */}
            <motion.div variants={itemVariants}>
              <h1 className="text-5xl lg:text-6xl xl:text-7xl font-black leading-[1.05] tracking-tight text-slate-900">
                Generate{" "}
                <span
                  className="relative inline-block"
                  style={{
                    background:
                      "linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #ec4899 80%)",
                    WebkitBackgroundClip: "text",
                    WebkitTextFillColor: "transparent",
                    backgroundClip: "text",
                  }}
                >
                  Stunning
                </span>
                <br />
                LinkedIn
                <br />
                <span className="text-slate-800">Portraits</span>{" "}
                <span className="text-slate-400 font-light">with AI</span>
              </h1>
            </motion.div>

            {/* Subtext */}
            <motion.p
              variants={itemVariants}
              className="text-lg lg:text-xl text-slate-500 leading-relaxed max-w-lg font-normal"
            >
              Describe any look. Our AI crafts the perfect professional portrait
              of{" "}
              <span className="font-semibold text-slate-700">Shl0k</span> in
              seconds — ready for LinkedIn, GitHub, or anywhere.
            </motion.p>

            {/* CTAs */}
            <motion.div
              variants={itemVariants}
              className="flex flex-wrap gap-3"
            >
              <motion.a
                href="#generator"
                onClick={scrollToGenerator}
                whileHover={{
                  scale: 1.03,
                  boxShadow:
                    "0 12px 32px rgba(99,102,241,0.35), 0 4px 12px rgba(139,92,246,0.2)",
                }}
                whileTap={{ scale: 0.97 }}
                transition={{ type: "spring", stiffness: 220, damping: 30 }}
                className="flex items-center gap-2.5 px-7 py-3.5 rounded-2xl font-semibold text-white text-base"
                style={{
                  background:
                    "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%)",
                  boxShadow:
                    "0 8px 24px rgba(99,102,241,0.25), 0 2px 8px rgba(139,92,246,0.15)",
                }}
              >
                Generate Now
                <ArrowRight className="w-4 h-4" />
              </motion.a>

              <motion.a
                href="#examples"
                onClick={(e) => {
                  e.preventDefault();
                  const el = document.getElementById("examples");
                  if (el) window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - 170, behavior: "smooth" });
                }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                transition={{ type: "spring", stiffness: 220, damping: 30 }}
                className="flex items-center gap-2 px-7 py-3.5 rounded-2xl font-semibold text-slate-600 text-base border border-slate-200/80 hover:border-slate-300 hover:bg-white/60 transition-all"
                style={{ backdropFilter: "blur(8px)" }}
              >
                See Examples
              </motion.a>
            </motion.div>

            {/* Social proof */}
            <motion.div
              variants={itemVariants}
              className="flex flex-wrap items-center gap-2 pt-2"
            >
              {["Groq", "FLUX.1-dev", "Modal"].map((tech, i) => (
                <span key={tech}>
                  <span
                    className="text-sm font-medium px-3 py-1 rounded-lg"
                    style={{
                      background: "rgba(248,247,255,0.8)",
                      border: "1px solid rgba(203,213,225,0.6)",
                      color: "#64748b",
                    }}
                  >
                    {tech}
                  </span>
                  {i < 2 && (
                    <span className="text-slate-300 mx-1 text-sm">·</span>
                  )}
                </span>
              ))}
            </motion.div>
          </motion.div>

          {/* ─── Right: Floating Cards ───────────────────────────────── */}
          <div className="relative h-[480px] lg:h-[560px] hidden lg:block">
            {/* Card 1 — back-left */}
            <div
              className="absolute"
              style={{ top: "8%", left: "4%", width: "200px", height: "280px" }}
            >
              <FloatingCard
                gradient={GRADIENT_PEACH}
                float={FLOAT_2}
                depth={2}
                label="Tech Professional"
              />
            </div>

            {/* Card 2 — front-right */}
            <div
              className="absolute"
              style={{
                top: "4%",
                right: "2%",
                width: "210px",
                height: "290px",
              }}
            >
              <FloatingCard
                gradient={GRADIENT_SKY}
                float={FLOAT_1}
                depth={1}
                label="Executive Style"
              />
            </div>

            {/* Card 3 — center main */}
            <div
              className="absolute"
              style={{
                top: "12%",
                left: "18%",
                width: "240px",
                height: "320px",
              }}
            >
              <FloatingCard
                gradient={GRADIENT_VIOLET}
                float={FLOAT_0}
                depth={0}
                label="LinkedIn Ready ✓"
              />
            </div>

            {/* Decorative glow blobs */}
            <div
              className="absolute rounded-full blur-3xl opacity-40 pointer-events-none"
              style={{
                width: "320px",
                height: "320px",
                background: "radial-gradient(circle, rgba(196,181,253,0.6), transparent 70%)",
                top: "20%",
                left: "10%",
              }}
            />
            <div
              className="absolute rounded-full blur-3xl opacity-30 pointer-events-none"
              style={{
                width: "260px",
                height: "260px",
                background:
                  "radial-gradient(circle, rgba(186,230,253,0.6), transparent 70%)",
                bottom: "10%",
                right: "5%",
              }}
            />
          </div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 0.8 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        >
          <motion.div
            animate={{ y: [0, 6, 0] }}
            transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
            className="w-5 h-8 rounded-full border-2 border-slate-300/60 flex items-start justify-center pt-1.5"
          >
            <div className="w-1 h-1.5 rounded-full bg-slate-400/60" />
          </motion.div>
          <span className="text-xs text-slate-400 font-medium">Scroll</span>
        </motion.div>
      </div>
    </section>
  );
}
